import torch
import numpy as np
import torch.nn.functional as F
import time
from dataclasses import asdict
from PIL import Image

from transformers import AutoTokenizer
from mmada_models import MMadaModelLM, MAGVITv2
from mmada_training.prompting_utils import UniversalPrompting
from dllm_cache.cache import dLLMCache, dLLMCacheConfig
from dllm_cache import register_cache_MMaDA, logout_cache_MMaDA

# Cache configuration parameters
prompt_interval_steps = 20  # Interval for refreshing prompt cache
gen_interval_steps = 5      # Interval for refreshing generation cache
transfer_ratio = 0.25       # Ratio of features to transfer from cache
use_cache = True            # Enable/disable dLLM-Cache

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    For MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Because MMaDA employs a linear noise schedule, the expected number of tokens 
    transitioned at each step should be consistent.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def mmu_generate_with_cache(model, input_ids=None, input_embeddings=None, max_new_tokens=128, steps=128,
                           block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0,
                           remasking='low_confidence', mask_id=126336, attention_mask=None):
    """
    Multimodal understanding generation with dLLM-Cache optimization.
    
    Args:
        model: MMaDA model
        input_ids: Input token ids (B, L)
        max_new_tokens: Number of tokens to generate
        steps: Number of sampling steps
        block_length: Block length for semi-autoregressive generation
        temperature: Sampling temperature
        remasking: Remasking strategy ('low_confidence' or 'random')
        mask_id: Mask token ID
        attention_mask: Attention mask
    """
    # Setup attention bias
    if attention_mask is not None and 0.0 in attention_mask:
        attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
    else:
        attention_bias = None
        
    # Get device
    try:
        device = input_ids.device
    except:
        device = input_embeddings.device

    # Initialize output tensor
    batch_size = input_ids.shape[0]
    x = torch.full((batch_size, input_ids.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(device)
    x[:, :input_ids.shape[1]] = input_ids.clone()
    
    # Track prompt indices
    prompt_index = (x != mask_id)

    # Prepare generation blocks
    assert max_new_tokens % block_length == 0
    num_blocks = max_new_tokens // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    # Generate tokens block by block
    for num_block in range(num_blocks):
        block_mask_index = (x[:, input_ids.shape[1] + num_block * block_length: input_ids.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        # Generate tokens step by step
        for i in range(steps):
            mask_index = (x == mask_id)
            
            # Handle classifier-free guidance if enabled
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                
                # 处理CFG情况
                if attention_bias is not None:
                    combined_attention_bias = torch.cat([attention_bias, attention_bias], dim=0)
                    logits = model(x_, attention_bias=combined_attention_bias).logits
                else:
                    logits = model(x_).logits
                    
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                # 标准前向传播，使用dLLM-Cache
                logits = model(x, attention_bias=attention_bias).logits

            # Apply Gumbel noise for sampling
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            # Apply remasking strategy
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # Prevent masking beyond current block
            x0_p[:, input_ids.shape[1] + (num_block + 1) * block_length:] = -np.inf

            # Update tokens
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            # Select tokens to update based on confidence
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def process_image(image_path, resolution=256):
    """Process image for MMaDA model input"""
    from torchvision import transforms
    
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return transform(image).unsqueeze(0)


def main():
    # Initialize device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load models and tokenizer
    model = MMadaModelLM.from_pretrained("Gen-Verse/MMaDA-8B-Base", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("Gen-Verse/MMaDA-8B-Base", trust_remote_code=True)
    vq_model = MAGVITv2().from_pretrained("showlab/magvitv2").to(device)
    
    # Initialize universal prompting
    uni_prompting = UniversalPrompting(
        tokenizer, 
        max_text_len=512, 
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
        ignore_id=-100, 
        cond_dropout_prob=0.1, 
        use_reserved_token=True
    )
    
    # Load and process image (replace with your image path)
    image_path = "./asset/test.jpg"  # Update with your image path
    processed_image = process_image(image_path).to(device)
    # Encode image with VQ model
    with torch.no_grad():
        _,image_tokens = vq_model.encode(processed_image)
    
    # Prepare prompt
    question = "Please describe this image in detail."
    
    # Prepare input for multimodal understanding
    prompt_text = [question]
    # input_data = (prompt_text, image_tokens)
    input_data = (image_tokens, prompt_text)
    input_ids, attention_mask,label_ids = uni_prompting(input_data, 'mmu')
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Initialize dLLM-Cache if enabled
    if use_cache:
        print("Testing with cache enabled")
        print(f"Cache settings: prompt_interval_steps={prompt_interval_steps}, gen_interval_steps={gen_interval_steps}, transfer_ratio={transfer_ratio},prompt length {input_ids.shape[1]}")
        
        # 初始化dLLM-Cache
        dLLMCache.new_instance(
            **asdict(
                dLLMCacheConfig(
                    prompt_interval_steps=prompt_interval_steps,
                    gen_interval_steps=gen_interval_steps,
                    transfer_ratio=transfer_ratio,
                )
            )
        )
        
        # 注册缓存钩子
        register_cache_MMaDA(model, "model.transformer.blocks")
        
        # 重要：必须在生成前设置prompt_length
        cache_instance = dLLMCache()
        cache_instance.reset_cache(prompt_length=input_ids.shape[1])
    else:
        print(f"Testing without cache,prompt length {input_ids.shape[1]}")
    
    # Generate text with timing
    start_time = time.time()
    
    output = mmu_generate_with_cache(
        model,
        input_ids=input_ids,
        max_new_tokens=256,
        steps=256,
        block_length=8,
        temperature=0,
        remasking='low_confidence',
        mask_id=126336,
        attention_mask=attention_mask
    )
    
    end_time = time.time()
    generation_time = end_time - start_time
    print(f"Generation time: {generation_time:.4f} seconds")
    # Decode and print output
    generated_text = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)
    print("Generated description:")
    print(generated_text)


if __name__ == '__main__':
    main() 