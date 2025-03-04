import torch
from transformers import AutoProcessor
from optimum.habana.transformers.models import GaudiQwen2VLForConditionalGeneration

#from qwen_vl_utils import process_vision_info
from PIL import Image
import requests
import os
import time
 
from optimum.habana.utils import set_seed
set_seed(42)
import habana_frameworks.torch.core as htcore
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
adapt_transformers_to_gaudi()
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
import json



def inference_by_krypton(image_path: str, vision_model, vision_processor, system_prompt: str, user_prompt: str):
    """Performs inference using the Krypton model on HPU."""
    try:
        generate_kwargs = {
            "lazy_mode": True,
            "hpu_graphs": True,
            "static_shapes": True,
            "use_cache": True,
            "cache_implementation": "static",
            "do_sample": False,
            "use_flash_attention": True
        }

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_prompt}
            ]}
        ]
        
        text_prompt = vision_processor.apply_chat_template(conversation, add_generation_prompt=True)
        image_tens = Image.open(image_path)
        inputs = vision_processor(text=[text_prompt], images=[image_tens], padding=True, return_tensors="pt")
        inputs = inputs.to("hpu")
        
        start_time = time.time()
        generated_ids = vision_model.generate(
            **inputs, max_new_tokens=8192, **generate_kwargs
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = vision_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        torch.hpu.synchronize()
        
        json_string = output_text[0]
        
        cleaned_json_string = json_string.strip().strip("```json").strip()
        data = json.loads(cleaned_json_string)
        for item in data:
            item["confidence"] = None
        
        return json.dumps(data)
    except Exception as ex:
        # logger.error(f"Exception occurred during Krypton inference for the file {image_path}: {ex}")
        raise ex





device = torch.device("hpu")

model = GaudiQwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16
        )
        
# Move model to HPU and optimize for HPU execution
model = model.to(device)
wrap_in_hpu_graph(model)

# Load processor with pixel constraints
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

min_pixels = 256*28*28
max_pixels = 2560*28*28
processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-72B-Instruct-AWQ',min_pixels=min_pixels,max_pixels=max_pixels)


img_path = 'SYNT_166730538_1_0.jpg'
user_pompt = open('user.txt','r').read()
system_prompt = open('system.txt','r').read()

for i in range(20):
    print(i)
    data = inference_by_krypton(img_path,model,processor,system_prompt,user_pompt)
    print(torch.hpu.memory_usage())
print(data)
