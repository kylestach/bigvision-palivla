import jax
import jax.numpy as jnp
from jax import lax
import re

import numpy as np
import tensorflow as tf
import wandb
from PIL import Image, ImageDraw

# get the next seven tokens after the begin of action token; otherwise, use zeros
def extract_action_tokens(step_out_tokens, begin_of_action_token):
    action_starts = (step_out_tokens == begin_of_action_token).astype(jnp.int32)
    first_action_start_idx = jnp.argmax(action_starts)+1 # first index of action token. 
    action_tokens = lax.cond(
        (jnp.all(action_starts == 0)) | (first_action_start_idx + 7 > step_out_tokens.shape[0]),
        lambda _: jnp.zeros(7, dtype=step_out_tokens.dtype),
        lambda _: lax.dynamic_slice(step_out_tokens, (first_action_start_idx,), (7,)),
        operand=None
    )
    return action_tokens

# get the cot tokens (tokens btwn begin CoT token and begin action token)
def extract_cot_strs(step_out_tokens, masked_prompt_tokens, beg_action_token, detokenize_lang_fn):

    # get prompt
    masked_prompt_tokens = np.array(masked_prompt_tokens)
    first_zero_idx = np.where(masked_prompt_tokens == 0)[0][0] if 0 in masked_prompt_tokens else len(masked_prompt_tokens)
    prompt_tokens = masked_prompt_tokens[:first_zero_idx]

    detokenize_prompt = detokenize_lang_fn(tf.convert_to_tensor(prompt_tokens, dtype=tf.int32))
    prompt_str = tf.strings.reduce_join(detokenize_prompt, separator="").numpy().decode("utf-8")

    # get cot
    step_out_tokens_np = np.array(step_out_tokens)

    try:
        action_start_idx = np.where(step_out_tokens_np == beg_action_token)[0][0]
    except IndexError:
        return prompt_str, "" 
    
    if action_start_idx<=0:
        return prompt_str, ""

    #the output sequence (step_tokens) directly starts from CoT, bc the prompt now includes the beg_cot_token
    cot_tokens = step_out_tokens_np[:action_start_idx]
    detokenized_cot = detokenize_lang_fn(tf.convert_to_tensor(cot_tokens, dtype=tf.int32))
    cot_str = tf.strings.reduce_join(detokenized_cot, separator="").numpy().decode("utf-8")

    return prompt_str, cot_str 

def viz_cot(image, lang_str, reasoning_str):
    matches = re.findall(r"gripper <loc(\d{4})><loc(\d{4})>", reasoning_str)
    gripper_plan = []
    for match in matches:
        y = float(match[0]) * 256 / 1024  # Normalize y
        x = float(match[1]) * 256 / 1024  # Normalize x
        gripper_plan.append([y, x])

    object_matches = re.findall(
        r"([a-zA-Z\s]+)<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>", reasoning_str
    )
    objects_bboxes = {}
    for obj_name, y_min, x_min, y_max, x_max in object_matches:
        name = obj_name.strip()
        bbox = [
            [int(y_min) * 256 / 1024, int(x_min) * 256 / 1024],  # Top-left corner
            [int(y_max) * 256 / 1024, int(x_max) * 256 / 1024],  # Bottom-right corner
        ]
        if name not in objects_bboxes:
            objects_bboxes[name] = []
        objects_bboxes[name].append(bbox)

    image = np.clip(image.squeeze() * 255, 0, 255).astype(np.uint8)
    obj_image = plot_obj_plans(image, lang_str, objects_bboxes)
    gripper_plan_image = plot_gripper_plan(image, lang_str, gripper_plan)

    return {
        "pred_obj_image": wandb.Image(obj_image),
        "pred_gripper_plan_image": wandb.Image(gripper_plan_image)
    }


def get_cot_table_metrics(lang_and_cot_strs):
    table = wandb.Table(columns=["GT Language Label", "Predicted CoT"])
    for lang_str, cot_str in lang_and_cot_strs:
        table.add_data(lang_str, cot_str)
    dct = {"CoT Outputs": table}
    return dct

def plot_obj_plans(image, lang_str, objects_bboxes):
    pil_image = Image.fromarray(image.squeeze())
    draw = ImageDraw.Draw(pil_image)
    
    # Overlay object bounding boxes for the current step
    for obj_name, bboxes in objects_bboxes.items():
        for (y_min, x_min), (y_max, x_max) in bboxes:
            draw.rectangle(
                [(x_min, y_min), (x_max, y_max)],
                outline=(0, 255, 0),
                width=2
            )
            draw.text((x_min + 5, y_min + 5), obj_name, fill=(255, 255, 255))
    
    draw.text((10, pil_image.height - 20), lang_str, fill=(255, 255, 255))
    
    return np.array(pil_image)

def plot_gripper_plan(image, lang_str, gripper_plan):
    pil_image = Image.fromarray(image.squeeze())
    draw = ImageDraw.Draw(pil_image)
    
    # Overlay the gripper plan
    for y,x in gripper_plan:
        # Draw a small circle for each gripper location
        draw.ellipse([(x - 4, y - 4), (x + 4, y + 4)], fill=(255, 0, 0))
    
    draw.text((10, pil_image.height - 20), lang_str, fill=(255, 255, 255))
    # Add the frame to the list
    return np.array(pil_image)