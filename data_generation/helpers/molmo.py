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

# make gpus visible

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
model.to(1)
print("model loaded **************")

# load the smaller processor
print("loading Molmo 7B .... **************")
processor2 = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    # device_map={"": "cuda:0"},
)

# load the model
model2 = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    # device_map={"": "cuda:0"},
    cache_dir="/global/scratch/users/riadoshi/cache/"
)
model2.to(2)
print("model loaded **************")

def remove_null(obj_names, obj_coords):
    new_obj_names, new_obj_coords = [],[]
    for name, coord in zip(obj_names, obj_coords):
        if coord[0] is not None and coord[1] is not None:
            new_obj_names.append(name)
            new_obj_coords.append(coord)
    return new_obj_names, new_obj_coords

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

def get_obj_coords(img, objs):
    img = Image.fromarray(img)
        
    coords = []

    for obj in objs: 
        prompt=f"point to the {obj}"
        inputs = processor2.process(images=img,text=prompt)
        inputs["images"] = inputs["images"].to(model2.device, dtype=torch.bfloat16)

        inputs = {k: v.to(model2.device).unsqueeze(0) for k, v in inputs.items()}

        output = model2.generate_from_batch(inputs,GenerationConfig(max_new_tokens=30, stop_strings=["alt", "none"]),tokenizer=processor2.tokenizer)

        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = processor2.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print("raw coords text: ", generated_text)

        # regex to find the point coordinates
        match = re.search(r'x="([-+]?[0-9]*\.?[0-9]+)"\s+y="([-+]?[0-9]*\.?[0-9]+)"', generated_text)

        if match:
            x, y = match.groups()
            x_scaled = float(x) / 100 * 256
            y_scaled = float(y) / 100 * 256
            coords.append((x_scaled, y_scaled))
        else:
            # if multiple points were generated, take the first one
            match = re.search(r'x1="([-+]?[0-9]*\.?[0-9]+)"\s+y1="([-+]?[0-9]*\.?[0-9]+)"', generated_text)
            if match:
                x, y = match.groups()
                x_scaled = float(x) / 100 * 256
                y_scaled = float(y) / 100 * 256
                coords.append((x_scaled, y_scaled))
            else:
                coords.append((None, None)) # default null value for now

    return coords

