import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images


# specify the path to the model
model_path = "deepseek-ai/deepseek-vl-7b-chat"
cache_dir = "/weka/scratch/tshu2/xli383/models"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path,cache_dir=cache_dir)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,cache_dir=cache_dir)
vl_gpt = vl_gpt.to(torch.float16).cuda(0).eval()

conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>Describe each stage of this image.",
        "images": ["/weka/scratch/tshu2/xli383/research/completion-infuse-robot/results/arrange_vegatable/demo_pick the vegetables from the sink and place them on the cutting board_demo_1/10.png"]
    },
    {
        "role": "Assistant",
        "content": ""
    }
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True,
    padding=True
).to(vl_gpt.device, dtype=torch.float16)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)
