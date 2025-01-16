import io
from typing import Any
from PIL import Image
import wandb

import jax
import jax.experimental.multihost_utils as mhu
import numpy as np

from PIL import Image, ImageDraw
import re

from big_vision.utils import Registry
from palivla.model_components import ModelComponents

def get_table(lang_str, pred_cot, target_cot):
    table = wandb.Table(columns=["Prompt", "Predicted CoT", "GT CoT"])
    table.add_data(lang_str, pred_cot, target_cot)
    return table

def plot_obj_plans(image, lang_str, objects_bboxes):
    pil_image = Image.fromarray(image.squeeze())
    draw = ImageDraw.Draw(pil_image)
    
    # Overlay object bounding boxes for the current step
    try:
        for obj_name, bboxes in objects_bboxes.items():
            for (y_min, x_min), (y_max, x_max) in bboxes:
                draw.rectangle(
                    [(x_min, y_min), (x_max, y_max)],
                    outline=(0, 255, 0),
                    width=2
                )
                draw.text((x_min + 5, y_min + 5), obj_name, fill=(255, 255, 255))
    except:
        draw.text((10, 10), "invalid bbox", fill=(255, 255, 255))
    
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


@Registry.register("viz.chain_of_thought")
def chain_of_thought(model: ModelComponents, trajectory: Any):
   frame = jax.tree.map(lambda x: x[:1], trajectory)
   frame["observation"] = jax.tree.map(lambda x: x[None], frame["observation"])
   frame["action"] = trajectory["action"][None, :1, :]

   # Predict chain-of-thought
   sequences = model.build_sequence(frame, begin_is_prompt=True)
   viz_batch, sequences = mhu.broadcast_one_to_all(({"observation": frame["observation"]}, sequences))
   predicted_tokens = model.predict_tokens(viz_batch, sequences, use_ema_params=True, replicate=True)

   # Decode the tokens
   predicted_text_tokens = np.array([model.language_tokenizer.decode(tok) for tok in predicted_tokens[0]])
   target_gen_tokens = sequences["gen"]["tokens"]
   target_gen_text_tokens = np.array([model.language_tokenizer.decode(tok) for tok in target_gen_tokens[0]])
   prompt_tokens = sequences["prompt"]["tokens"]
   prompt_text_tokens = np.array([model.language_tokenizer.decode(tok) for tok in prompt_tokens[0] if tok != '<pad>'])

   # Get action start indices
   pred_action_start_idxs = np.argwhere(predicted_text_tokens == '<begin_of_action>') + 1
   pred_action_start_idx = np.min(pred_action_start_idxs) if pred_action_start_idxs.size > 0 else -1
   target_action_start_idx = np.min(np.argwhere(target_gen_text_tokens == '<begin_of_action>')) + 1

   # Concatenate tokens into strings
   prompt_str = ''.join(prompt_text_tokens)
   pred_cot_str = ''.join(predicted_text_tokens[:pred_action_start_idx])
   target_cot_str = ''.join(target_gen_text_tokens[:target_action_start_idx])

   # Plot the table of prompt, target reasoning str, gen reasoning str
   cot_table = get_table(prompt_str, pred_cot_str, target_cot_str)

   # Make plots of predicted CoT
   matches = re.findall(r"gripper <loc(\d{4})><loc(\d{4})>", pred_cot_str)
   gripper_plan = []
   for match in matches:
      y = float(match[0]) * 224 / 1024  # Normalize y
      x = float(match[1]) * 224 / 1024  # Normalize x
      gripper_plan.append([y, x])

   object_matches = re.findall(
      r"([a-zA-Z\s]+)<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>", pred_cot_str
   )
   objects_bboxes = {}
   for obj_name, y_min, x_min, y_max, x_max in object_matches:
      name = obj_name.strip()
      bbox = [
            [int(y_min) * 224 / 1024, int(x_min) * 224 / 1024],  # Top-left corner
            [int(y_max) * 224 / 1024, int(x_max) * 224 / 1024],  # Bottom-right corner
      ]
      if name not in objects_bboxes:
            objects_bboxes[name] = []
      objects_bboxes[name].append(bbox)

   image = frame["observation"]['image_primary'].squeeze()
   obj_image = plot_obj_plans(image, prompt_str, objects_bboxes)
   gripper_plan_image = plot_gripper_plan(image, prompt_str, gripper_plan)

   return {
      "pred_obj_image": wandb.Image(obj_image),
      "pred_gripper_plan_image": wandb.Image(gripper_plan_image),
      "cot_outputs": cot_table,
   }


