import torch
import numpy as np
import torch.nn.functional as F
import time
from dataclasses import asdict

from transformers import AutoTokenizer
from mmada_models import MMadaModelLM
from dllm_cache.cache import dLLMCache, dLLMCacheConfig
from dllm_cache import register_cache_MMaDA, logout_cache_MMaDA

# Cache configuration parameters
prompt_interval_steps = 20  # Interval for refreshing prompt cache
gen_interval_steps = 2     # Interval for refreshing generation cache
transfer_ratio = 0.25       # Ratio of features to transfer from cache
use_cache = True            # Enable/disable dLLM-Cache

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    For MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    '''
    if temperature == 0:
        return logits
    noise = torch.rand_like(logits)
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
def generate_with_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                       cfg_scale=0., remasking='low_confidence', mask_id=126336, attention_mask=None):
    if attention_mask is not None and 0.0 in attention_mask:
        attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
    else:
        attention_bias = None
    
    batch_size = 1
    x = torch.full((batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                
                # CFG情况下
                if attention_bias is not None:
                    combined_attention_bias = torch.cat([attention_bias, attention_bias], dim=0)
                    output = model(x_, attention_bias=combined_attention_bias)
                else:
                    output = model(x_)
                    
                logits = output.logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                # 正常生成情况下，使用dLLM-Cache
                output = model(x, attention_bias=attention_bias)
                logits = output.logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def main():
    # Initialize device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model and tokenizer
    model = MMadaModelLM.from_pretrained("Gen-Verse/MMaDA-8B-Base", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("Gen-Verse/MMaDA-8B-Base", trust_remote_code=True)
    tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}"
    
    # Prepare input
    # prompt = "Question: After receiving the $2000 stimulus check, Mr. Eithan decided to share the amount with his family. He gave 2/5 of the amount to his wife, 2/5 of the remaining amount to his first son, 40% of the remaining amount to his second son, and kept the remaining in their family savings account. Calculate the total amount he kept in the family's savings account.\nAnswer: The total amount of money Mr. Eithan gave to his wife is 2/5*2000 = $<<2/5*2000=800>>800\nAfter giving his wife $800, he remained with $2000-$800=$<<2000-800=1200>>1200\nHe gave his first son 2/5 of the remaining amount which is 2/5*$1200 = $<<2/5*1200=480>>480\nThe total amount remaining after he gave his first 2/5 of the amount is $1200-$480 = $<<1200-480=720>>720\nHe then gave his second son 40/100*720 = $<<40/100*720=288>>288 of the money.\nAfter giving his second son $288, the amount of money remaining that he saved in the family's saving account is $720-$288=$432\n#### 432\n\nQuestion: Roosevelt High school plays a basketball tournament with Greendale High school. Roosevelt high school scores 30 points in the first game, half as much in the second game, and triple as much as the second game in the third game. At the end of the tournament, Roosevelt high school receives 50 bonus points and Greendale high school has 10 points less than Roosevelt high school. How many points does Greendale high school have?\nAnswer: The points Roosevelt high school has for the second game are 30/2=<<30/2=15>>15 points.\nThe points Roosevelt high school has for the third game are 15*3=<<15*3=45>>45 points.\nThe total points Roosevelt high school has for the tournament are 30+15+45+50=<<30+15+45+50=140>>140 points.\nThe total points Greendale high school has for the tournament are 140-10=<<140-10=130>>130 points.\n#### 130\n\nQuestion: On Tuesday, a fruit vendor sold 2.5 dozen lemons and 5 dozens avocados. What is the total number of fruits that the fruit vendor sold?\nAnswer: Since 1 dozen is equal to 12, then the vendor sold 2.5 x 12 = <<2.5*12=30>>30 lemons.\nWhile he sold 5 x 12 = <<5*12=60>>60 avocados.\nSo, the fruit vendor sold a total of 30 + 60 = <<30+60=90>>90 fruits.\n#### 90\n\nQuestion: Sandra wants to buy some sweets. She saved $10 for this purpose. Her mother gave her an additional $4, and her father twice as much as her mother. One candy costs $0.5, and one jelly bean $0.2. She wants to buy 14 candies and 20 jelly beans. How much money will she be left with after the purchase?\nAnswer: Sandra's father gave her $4 * 2 = $<<4*2=8>>8.\nSo Sandra has in total $8 + $4 + $10 = $<<8+4+10=22>>22.\nShe wants 14 candies, so she is going to pay 14 candies * $0.50/candy = $<<14*0.5=7>>7 for them.\nShe wants also 20 jellybeans, and they're going to cost 20 jellybeans * $0.20/jellybean = $<<20*0.2=4>>4.\nSo after the purchase, she will be left with $22 - $4 - $7 = $<<22-4-7=11>>11.\n#### 11\n\nQuestion: Tracy used a piece of wire 4 feet long to support tomato plants in the garden. The wire was cut into pieces 6 inches long. How many pieces did she obtain?\nAnswer:"
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
    # m = [{"role": "user", "content": prompt}, ]
    # prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(text=prompt, return_tensors="pt", padding=True, padding_side="left")['input_ids']
    input_ids = input_ids.to(device)
    
    # Initialize dLLM-Cache if enabled
    if use_cache:
        print("Testing with cache enabled")
        print(f"Cache settings: prompt_interval_steps={prompt_interval_steps}, gen_interval_steps={gen_interval_steps}, transfer_ratio={transfer_ratio},prompt length is {input_ids.shape[1]}")
        
        dLLMCache.new_instance(
            **asdict(
                dLLMCacheConfig(
                    prompt_interval_steps=prompt_interval_steps,
                    gen_interval_steps=gen_interval_steps,
                    transfer_ratio=transfer_ratio,
                )
            )
        )
        # 初始化cache
        cache_instance = dLLMCache()
        cache_instance.reset_cache(prompt_length=input_ids.shape[1])
        register_cache_MMaDA(model, "model.transformer.blocks")
    else:
        print(f"Testing without cache,prompt length is {input_ids.shape[1]}")
    
    # Generate text with timing
    start_time = time.time()
    out = generate_with_cache(
        model, 
        input_ids, 
        steps=256, 
        gen_length=256, 
        block_length=32, 
        temperature=0, 
        cfg_scale=0., 
        remasking='low_confidence'
    )
    end_time = time.time()
    
    # Print results
    generation_time = end_time - start_time
    print(f"Generation time: {generation_time:.4f} seconds")
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True))


if __name__ == '__main__':
    main()