## run molmo VLM on a bunch of saved trajectories

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import matplotlib.pyplot as plt
import mediapy
import os
import numpy as np
import xml.etree.ElementTree as ET
import torch
import re

# load the processor
device="auto"
processor = AutoProcessor.from_pretrained(
    'allenai/MolmoE-1B-0924',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    # device_map={"": "cuda:1"},
)

# load the model
print("loading Molmo MoE .... **************")
model = AutoModelForCausalLM.from_pretrained(
    'allenai/MolmoE-1B-0924',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    # device_map={"": "cuda:1"},
    cache_dir="/global/scratch/users/riadoshi/cache"
)
model.to(0)
print("model loaded **************")


def get_obj_names(img, lang):
 
    img = Image.fromarray(img)

    PROMPT=f"The robot task is {lang}. Briefly provide a list of up to 4 objects in the scene (ignore the robot), including the ones in the task. don't add any additional text:"
    inputs = processor.process(images=img,text=PROMPT)
    inputs["images"] = inputs["images"].to(model.device, dtype=torch.bfloat16)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    output = model.generate_from_batch(inputs,GenerationConfig(max_new_tokens=45, stop_strings="<|endoftext|>"),tokenizer=processor.tokenizer)
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    traj_objs = generated_text.split('\n')
    final_objs = [obj[3:].strip() for obj in traj_objs] # generates it as "1. obj1 2. obj2 etc, so doing [3:] just removes the number associated w each object"
    return final_objs


# def get_obj_names_batched(imgs, langs):
#     inputs = []
#     for img, lang in zip(imgs, langs):
#         print(img.shape)

#         img = Image.fromarray(img)

#         PROMPT=f"The robot task is {lang}. Briefly provide a list of up to 4 objects in the scene (ignore the robot), including the ones in the task. don't add any additional text:"
#         inputs.append(processor.process(images=img,text=PROMPT, padding=True))
    
#     for inp in inputs:
#         inp["images"] = inp["images"].to(model.device, dtype=torch.bfloat16)
#         inp = {k: v.to(model.device).unsqueeze(0) for k, v in inp.items()}
    
#     input_combined = {k: torch.stack([inp[k] for inp in inputs]).to(model.device) for k in inputs[0]}

#     output = model.generate_from_batch(input_combined,GenerationConfig(max_new_tokens=45, stop_strings="<|endoftext|>"),tokenizer=processor.tokenizer)
    
#     objs_list = []
#     for i in range(output.size(0)):
#         generated_tokens = output[i,inputs['input_ids'].size(1):]
#         generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

#         traj_objs = generated_text.split('\n')
#         objs_list.append([obj[3:].strip() for obj in traj_objs]) # generates it as "1. obj1 2. obj2 etc, so doing [3:] just removes the number associated w each object"
#     return objs_list