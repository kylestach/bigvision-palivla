import tensorflow as tf
import numpy as np

from PIL import Image, ImageDraw
import mediapy
import re

def plotted_gripper(traj):
    gripper_plans = []
    for reasoning in traj['reasonings']:
        # Find all occurrences of "gripper" followed by loc tokens
        reasoning = reasoning.decode('utf-8')
        matches = re.findall(r"gripper <loc(\d{4})><loc(\d{4})>", reasoning)
        gripper_plan = []
        
        for match in matches:
            # Extract the integers and normalize as specified
            y = float(match[0]) * 256 / 1024  # Normalize y
            x = float(match[1]) * 256 / 1024  # Normalize x
            gripper_plan.append([y, x])
        
        # Append the list of locations for this step
        gripper_plans.append(gripper_plan)

    return gripper_plans

def plot_gripper_plans(traj, step=5):
    gripper_plans = plotted_gripper(traj)
    observations = traj['observation']['image_primary'][step:step+5]
    grip_plan = gripper_plans[step:step+5][0]
    print(grip_plan)
    
    overlayed_frames = []
    for idx, (image, _) in enumerate(zip(observations, gripper_plans[step:step+5])):
        # Convert the image to a PIL format
        pil_image = Image.fromarray(image.squeeze())
        draw = ImageDraw.Draw(pil_image)
        
        # Overlay the gripper plan
        for y,x in grip_plan:
            # Draw a small circle for each gripper location
            draw.ellipse([(x - 4, y - 4), (x + 4, y + 4)], fill=(255, 0, 0))
        
        # Add the frame to the list
        overlayed_frames.append(np.array(pil_image))
    
    # Display the frames as a video using mediapy
    mediapy.show_video(overlayed_frames, fps=1)  # Slower FPS for clarity


def parse_object_bounding_boxes(reasoning):
    """
    Extract object bounding boxes from a reasoning string.
    """
    reasoning_str = reasoning.decode("utf-8")  # Decode byte string
    object_matches = re.findall(
        r"([a-zA-Z\s]+)<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>", reasoning_str
    )
    objects = {}
    for obj_name, y_min, x_min, y_max, x_max in object_matches:
        name = obj_name.strip()
        bbox = [
            [int(y_min) * 256 / 1024, int(x_min) * 256 / 1024],  # Top-left corner
            [int(y_max) * 256 / 1024, int(x_max) * 256 / 1024],  # Bottom-right corner
        ]
        if name not in objects:
            objects[name] = []
        objects[name].append(bbox)
    return objects

def plot_object_plans(traj, step=0):
    """
    Plot gripper and object plans over observations for the next 5 steps.
    """
    # Parse gripper and object plans
    object_bboxes = [
        parse_object_bounding_boxes(reasoning) for reasoning in traj['reasonings']
    ]
    
    # Extract the observations for the specified steps
    observations = traj['observation']['image_primary'][step:step+5]
    
    # Initialize a list to hold the frames
    overlayed_frames = []
    
    for idx, image in enumerate(observations):
        step_idx = step + idx
        pil_image = Image.fromarray(image.squeeze())
        draw = ImageDraw.Draw(pil_image)
        
        # Overlay object bounding boxes for the current step
        for obj_name, bboxes in object_bboxes[step].items():
            for (y_min, x_min), (y_max, x_max) in bboxes:
                draw.rectangle(
                    [(x_min, y_min), (x_max, y_max)],
                    outline=(0, 255, 0),
                    width=2
                )
                draw.text((x_min + 5, y_min + 5), obj_name, fill=(255, 255, 255))
        
        # Add the frame to the list
        overlayed_frames.append(np.array(pil_image))
    
    # Display the frames as a video using mediapy
    mediapy.show_video(overlayed_frames, fps=1)  # Slower FPS for clarity