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

# load the smaller processor
print("loading Molmo 7B .... **************")
processor2 = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# load the model
model2 = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    cache_dir="/global/scratch/users/riadoshi/cache/"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2.to(device)
print("model loaded **************")


def remove_null(obj_names, obj_coords):
    new_obj_names, new_obj_coords = [],[]
    for name, coord in zip(obj_names, obj_coords):
        if coord[0] is not None and coord[1] is not None:
            new_obj_names.append(name)
            new_obj_coords.append(coord)
    return new_obj_names, new_obj_coords


def get_obj_coords(img, objs):
    img = Image.fromarray(img)
        
    coords = []

    for obj in objs: 
        with torch.no_grad():
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
    torch.cuda.empty_cache()

    return coords


