'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import argparse
import accelerate
import torch
import re
import sys
import os
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from dataclasses import asdict

from transformers import AutoTokenizer, AutoModel

# --- Import dLLM-Cache and profiled_generate ---
cur_dir = os.path.dirname(os.path.abspath(__file__))
dllm_cache_dir = os.path.join(cur_dir, '..', 'dLLM-cache')
sys.path.append(dllm_cache_dir)

from dllm_cache.cache import dLLMCache, dLLMCacheConfig
from dllm_cache.hooks import register_cache_LLaDA
from my_utils.run_inference import profiled_generate
# --------------------------------------- --------

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        device="cuda",
        bnb_4bit_compute_dtype="bfloat16",
        # --- New arguments for dLLM-Cache ---
        enable_dllm_cache=False,
        cache_prompt_interval=100,
        cache_gen_interval=7,
        cache_transfer_ratio=0.25,
        # ------------------------------------
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
            enable_dllm_cache: Enable dLLM-Cache for generation tasks.
            cache_prompt_interval: Prompt refresh interval.
            cache_gen_interval: Generation refresh interval.
            cache_transfer_ratio: Ratio of tokens to transfer.
        '''
        super().__init__()

        # Allow code eval
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

        # Fixed cache settings (consistent with notebook)
        os.environ['HF_HOME'] = '/root/LLaDA/hf_models/'
        _MODEL_ID = model_path if model_path else 'GSAI-ML/LLaDA-8B-Instruct'
        _CACHE_DIR = '/root/LLaDA/hf_models/hub'

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        # Load model/tokenizer from fixed cache
        self.model = AutoModel.from_pretrained(
            _MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            cache_dir=_CACHE_DIR,
            local_files_only=True,
            **model_kwargs
        )
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_ID,
            trust_remote_code=True,
            cache_dir=_CACHE_DIR,
            local_files_only=True,
        )

        self.mc_num = int(mc_num)
        self.batch_size = int(batch_size)
        assert self.mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = str(is_check_greedy).lower() == 'true'

        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking

        # Initialize dLLM-Cache
        self.enable_dllm_cache = str(enable_dllm_cache).lower() == 'true'
        if self.enable_dllm_cache:
            cache_config = dLLMCacheConfig(
                prompt_interval_steps=int(cache_prompt_interval),
                gen_interval_steps=int(cache_gen_interval),
                transfer_ratio=float(cache_transfer_ratio)
            )
            dLLMCache.new_instance(**asdict(cache_config))
            
            # Register hooks (assuming standard LLaDA structure)
            register_cache_LLaDA(self.model, "model.transformer.blocks")
            print(f"dLLM-Cache enabled with config: {cache_config}")

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096, f"Prompt length exceeding 4096."

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        for elem in tqdm(ds, desc="Generating..."):
            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]

            # Reset dLLM-Cache
            if self.enable_dllm_cache:
                cache = dLLMCache()
                cache.reset_cache(prompt.shape[1])
                cache.step_logs = {}

            # Use profiled_generate from run_inference.py (single pass)
            generated_tensor, _ = profiled_generate(
                self.model,
                prompt,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=0,
                cfg_scale=self.cfg,
                remasking=self.remasking,
                mask_id=self.mask_id,
            )

            generated_answer = self.tokenizer.decode(generated_tensor[0][prompt.shape[1]:], skip_special_tokens=False)
            for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            out.append(generated_answer)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out


def build_model_args_string(parsed):
    # Build a comma-separated model_args string for lm-eval
    pairs = {
        'model_path': parsed.model_path,
        'mask_id': parsed.mask_id,
        'max_length': parsed.max_length,
        # 'batch_size': parsed.batch_size,
        'mc_num': parsed.mc_num,
        'is_check_greedy': parsed.is_check_greedy,
        'cfg': parsed.cfg,
        'steps': parsed.steps,
        'gen_length': parsed.gen_length,
        'block_length': parsed.block_length,
        'remasking': parsed.remasking,
        'device': parsed.device,
        'enable_dllm_cache': parsed.enable_dllm_cache,
        'cache_prompt_interval': parsed.cache_prompt_interval,
        'cache_gen_interval': parsed.cache_gen_interval,
        'cache_transfer_ratio': parsed.cache_transfer_ratio,
    }
    return ','.join([f"{k}={v}" for k, v in pairs.items()])


def parse_cli():
    parser = argparse.ArgumentParser(description="Evaluate LLaDA with lm-eval-harness using profiled_generate.")
    parser.add_argument('--tasks', type=str, default='humaneval')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--mc_num', type=int, default=128)
    parser.add_argument('--mask_id', type=int, default=126336)
    parser.add_argument('--max_length', type=int, default=4096)
    parser.add_argument('--is_check_greedy', type=str, default='false')
    parser.add_argument('--cfg', type=float, default=0.0)
    parser.add_argument('--steps', type=int, default=1024)
    parser.add_argument('--gen_length', type=int, default=1024)
    parser.add_argument('--block_length', type=int, default=1024)
    parser.add_argument('--remasking', type=str, default='low_confidence')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--model_args', type=str, default='')
    # dLLM-Cache
    parser.add_argument('--enable_dllm_cache', type=str, default='false')
    parser.add_argument('--cache_prompt_interval', type=int, default=100)
    parser.add_argument('--cache_gen_interval', type=int, default=7)
    parser.add_argument('--cache_transfer_ratio', type=float, default=0.25)
    # lm-eval passthrough
    parser.add_argument('--other_args', type=str, default='', help='Additional lm-eval args, e.g., "--limit 10"')
    return parser.parse_args()


if __name__ == "__main__":
    set_seed(1234)
    parsed = parse_cli()

    # Parse model_args to override defaults
    if parsed.model_args:
        for pair in parsed.model_args.split(','):
            if '=' in pair:
                k, v = pair.split('=', 1)
                k = k.strip()
                v = v.strip()
                if v.startswith("'") and v.endswith("'"): v = v[1:-1]
                if hasattr(parsed, k):
                    t = type(getattr(parsed, k))
                    try:
                        setattr(parsed, k, t(v))
                    except ValueError:
                        setattr(parsed, k, v)
                else:
                    setattr(parsed, k, v)

    model_args_str = build_model_args_string(parsed)

    cli_argv = [
        sys.argv[0],
        "--model", "llada_dist",
        "--tasks", parsed.tasks,
        "--batch_size", str(parsed.batch_size),
        "--model_args", model_args_str,
    ]

    if parsed.other_args:
        cli_argv.extend(parsed.other_args.split())

    sys.argv = cli_argv
    cli_evaluate()
