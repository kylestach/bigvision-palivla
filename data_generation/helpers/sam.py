
## run molmo VLM on a bunch of saved trajectories

from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from sam2.build_sam import build_sam2_video_predictor

import torch

torch.autocast(device_type='cuda', dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

######################################################################################
SAM2_CHECKPOINT = "/global/scratch/users/riadoshi/cache/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

PREDICTOR = build_sam2_video_predictor(MODEL_CFG, SAM2_CHECKPOINT)

def get_masks(object_coords, images):
    

    # save imgs to directory as jpgs
    ## BRC TODO: maybe change this temp directory?
    jpeg_path = f'/tmp/jpegs/'

    # jpeg conversion
    if not os.path.isdir(jpeg_path):
        os.mkdir(jpeg_path)
    for i, im_arr in enumerate(images):
        if not os.path.exists(jpeg_path):
            os.mkdir(jpeg_path) 
        Image.fromarray(im_arr).save(f'{jpeg_path}/{i}.jpeg')
    frame_names = [
        p for p in os.listdir(jpeg_path)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # initialize model
    inference_state = PREDICTOR.init_state(video_path=jpeg_path)
    PREDICTOR.reset_state(inference_state)

    # give a unique id to each object we interact with (it can be any integers)
    ann_obj_ids = list(range(0, len(object_coords)))  
    points = [np.array([point], dtype=np.float32) for point in object_coords]

    # if no points, return None
    if len(points) == 0:
        return None 
    
    # for labels, `1` means positive click and `0` means negative click
    for pi, point in enumerate(points):
        label = np.array([1], np.int32)
        _, out_obj_ids, out_mask_logits = PREDICTOR.add_new_points(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=ann_obj_ids[pi],
            points=point,
            labels=label,
        )

    print("points received by SAM! propagating through trajectory")

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in PREDICTOR.propagate_in_video(inference_state, start_frame_idx=0):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    ## RM DIR 
    os.system(f'rm -rf {jpeg_path}')

    torch.cuda.empty_cache()
    
    return video_segments