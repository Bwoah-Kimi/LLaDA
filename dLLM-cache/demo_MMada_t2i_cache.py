import torch
import numpy as np
import torch.nn.functional as F
import time
from dataclasses import asdict
from PIL import Image

from transformers import AutoTokenizer
from mmada_models import MMadaModelLM, MAGVITv2
from mmada_training.prompting_utils import UniversalPrompting
from mmada_models.sampling import cosine_schedule,mask_by_random_topk
from dllm_cache.cache import dLLMCache, dLLMCacheConfig
from dllm_cache import register_cache_MMaDA, logout_cache_MMaDA

# Cache configuration parameters
prompt_interval_steps = 10  # Interval for refreshing prompt cache
gen_interval_steps = 1      # Interval for refreshing generation cache
transfer_ratio = 0.0       # Ratio of features to transfer from cache
use_cache = True            # Enable/disable dLLM-Cache



@torch.no_grad()
def t2i_generate_with_cache(
        model,
        input_ids: torch.LongTensor = None,
        uncond_input_ids: torch.LongTensor = None,
        attention_mask=None,
        uncond_attention_mask=None,
        temperature=1.0,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        guidance_scale=0,
        noise_schedule=cosine_schedule,
        generator: torch.Generator = None,
        seq_len=1024,
        mask_token_id=126336,
        vq_model=None,
        uni_prompting=None,
):
    """
    Text-to-image generation with dLLM-Cache optimization.
    """
    # Calculate mask count and prepare variables
    mask_count = (input_ids == mask_token_id).sum().item()
    num_vq_tokens = seq_len
    num_new_special_tokens = 0
    codebook_size = 8192
    
    # Process input ids
    input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
    input_ids_minus_lm_vocab_size = torch.where(
        input_ids_minus_lm_vocab_size == mask_token_id, 
        mask_token_id, 
        input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens
    )

    # For classifier-free guidance
    if uncond_input_ids is not None:
        uncond_prefix = uncond_input_ids[:, :512 + 1]  # Adjust based on resolution

    # Main generation loop
    for step in range(timesteps):
        if uncond_input_ids is not None and guidance_scale > 0:
            # Handle classifier-free guidance
            uncond_input_ids = torch.cat(
                [uncond_prefix, input_ids[:, 512 + 1:]], dim=1)
            model_input = torch.cat([input_ids, uncond_input_ids])
            attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            
            # Get logits with guidance
            logits = model(model_input, attention_bias=attention_bias).logits
            cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
            logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
            
            # Extract relevant logits for image tokens
            logits = logits[:, -(num_vq_tokens + 1):-1, 
                    len(uni_prompting.text_tokenizer) + num_new_special_tokens: 
                    len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
        else:
            # Standard forward pass
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            logits = model(input_ids, attention_bias=attention_bias).logits
            
            # Extract relevant logits for image tokens
            logits = logits[:, -(num_vq_tokens + 1):-1, 
                    len(uni_prompting.text_tokenizer) + num_new_special_tokens: 
                    len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]

        # Sample from logits
        probs = logits.softmax(dim=-1)
        sampled = probs.reshape(-1, logits.size(-1))
        sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

        # Update tokens based on mask
        unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
        sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
        
        # Calculate mask ratio for next round
        ratio = 1.0 * (step + 1) / timesteps
        mask_ratio = noise_schedule(torch.tensor(ratio))
        
        # Get probabilities of selected tokens
        selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
        selected_probs = selected_probs.squeeze(-1)

        # Ignore tokens given in the input
        selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
        
        # Calculate mask length for next iteration
        mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
        mask_len = torch.max(
            torch.tensor([1], device=logits.device), 
            torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
        )
        
        # Add noise for randomness
        temperature_step = temperature * (1.0 - ratio)
        masking = mask_by_random_topk(mask_len, selected_probs, temperature_step, generator=generator)
        
        # Update input ids with new tokens
        input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(
            masking, 
            mask_token_id,
            sampled_ids + len(uni_prompting.text_tokenizer) + num_new_special_tokens
        )
        input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

    # Convert sampled ids to image using VQ model
    if vq_model is not None:
        with torch.no_grad():
            # image = vq_model.decode_code(sampled_ids.unsqueeze(0))
            print("sampled_ids.shape",sampled_ids.shape)
            # print(vq_model.decode_code(sampled_ids))
            image = vq_model.decode_code(sampled_ids)
            image = (image + 1) / 2  # Normalize to [0, 1]
            image = torch.clamp(image, 0, 1)
            image = (image * 255).to(torch.uint8)
            image = image[0].permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
            return image
    
    return sampled_ids


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
    
    # Prepare input
    prompt_text = ["A beautiful sunset over the mountains with a lake in the foreground"]
    image_tokens = torch.ones((1, 1024), dtype=torch.long, device=device) * 126336  # mask_id
    input_ids, attention_mask = uni_prompting((prompt_text, image_tokens), 't2i_gen')
    
    # For classifier-free guidance
    uncond_input_ids, uncond_attention_mask = uni_prompting(([''], image_tokens), 't2i_gen')
    
    # Initialize dLLM-Cache if enabled
    if use_cache:
        print("Testing with cache enabled")
        print(f"Cache settings: prompt_interval_steps={prompt_interval_steps}, gen_interval_steps={gen_interval_steps}, transfer_ratio={transfer_ratio}")
        
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
        print("Testing without cache")
    
    # Generate image with timing
    start_time = time.time()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    generator = torch.Generator(device=device).manual_seed(42)
    
    image = t2i_generate_with_cache(
        model,
        input_ids=input_ids,
        uncond_input_ids=uncond_input_ids,
        attention_mask=attention_mask,
        uncond_attention_mask=uncond_attention_mask,
        temperature=1.0,
        timesteps=18,
        guidance_scale=3.5,
        noise_schedule=cosine_schedule,
        generator=generator,
        seq_len=1024,
        mask_token_id=126336,
        vq_model=vq_model,
        uni_prompting=uni_prompting,
    )
    
    end_time = time.time()
    generation_time = end_time - start_time
    print(f"Generation time: {generation_time:.4f} seconds")
    image.save("generated_image_with_cache.png")
    print("Image saved as 'generated_image_with_cache.png'")


if __name__ == '__main__':
    main() 