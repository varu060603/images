# import torch
# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# #from qwen_vl_utils import process_vision_info
# from PIL import Image
# import requests
# import os
# from time import time
 
# from optimum.habana.utils import set_seed
# set_seed(42)
# import habana_frameworks.torch.core as htcore
# from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
# adapt_transformers_to_gaudi()
# from habana_frameworks.torch.hpu import wrap_in_hpu_graph
 
# # from huggingface_hub import login
# # login('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')
 
 
# model_name = 'Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4'
 
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     model_name, torch_dtype=torch.bfloat16
# )
# print('1')
# device = torch.device('hpu')
# model = model.to(device)
# print('2')
# wrap_in_hpu_graph(model)
# print('3')

# # from transformers import AutoConfig
# # config = AutoConfig.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")
# # for lyr in range(config.num_hidden_layers):
# #     wrap_in_hpu_graph(model.model.layers[lyr])
 
# min_pixels = 256*28*28
# max_pixels = 2560*28*28
# processor = AutoProcessor.from_pretrained(model_name,min_pixels=min_pixels,max_pixels=max_pixels)
 
# images = [Image.open('/root/akshit/qwen_test_data/images/SYNT_166730538_1_0.jpg')]
# # image_path = "qwen2-dataset/images/"
# # image_files = os.listdir(image_path)
# # for i in range(len(image_files)):
# #     images.append(Image.open(image_path+image_files[i]))
 
# prompts = []
# prompt_path = "/root/akshit/qwen_test_data/new_prompts/"
 
# prompt_files = os.listdir(prompt_path)
# for i in range(len(prompt_files)):
#     f = open(prompt_path+prompt_files[i], "r")
#     prompts.append(f.read())
#     #print(prompts[i])
#     f.close()
 
# max_new_tokens = 256
 
# generate_kwargs = {
#     "lazy_mode": True,
#     "use_cache": True,
#     "cache_implementation": "static",
#     "do_sample": False,
# }
 
 
# for prompt in prompts:
#     conversation = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                 },
#                 {"type": "text", "text": prompt},
#             ],
#         }
#     ]
 
   
#     text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
 
#     inputs = processor(
#         text=[text_prompt], images=[images[0]], padding=True, return_tensors="pt"
#     )
 
#     # To HPU
#     inputs = inputs.to(device)
 
#     t1 = time()
#     print('4')
#     generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, **generate_kwargs)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
#     print('5')
#     t2 = time()
#     print(output_text)






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


img_path = '/root/akshit/qwen_test_data/images/SYNT_166730538_1_3.jpg'
user_pompt = open('/root/akshit/qwen_test_data/new_prompts/prompt1.txt','r').read()
system_prompt = open('/root/akshit/qwen_test_data/new_prompts/prompt2.txt','r').read()

for i in range(20):
    print(i)
    data = inference_by_krypton(img_path,model,processor,system_prompt,user_pompt)
    print(torch.hpu.memory_usage())
print(data)