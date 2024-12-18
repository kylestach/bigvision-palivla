import numpy as np
from PIL import Image 
import io 
import tensorflow as tf
import argparse
import os
import json
import sys
sys.path.insert(0, '/global/scratch/users/riadoshi/bigvision-palivla/')

from big_vision.pp.proj.paligemma.segmentation import encode_to_codebook_indices, get_checkpoint

VQVAE_CKPT = dict(np.load('/global/scratch/users/riadoshi/cache/vae-oid.npz'))

def remove_irrelevant_objs(obj_id_to_name, language_label):
    def is_relevant(obj_name, language_label):
        words_in_obj_name = obj_name.split(" ")
        
        # at least one word from the object name should be present in the language label
        if any([True if word in language_label else False for word in words_in_obj_name]):
            return True
        return False

    relevant_ids = [] 
    for obj_id, name in obj_id_to_name.items():
        # heuristic 
        if is_relevant(name, language_label):
            relevant_ids.append(obj_id)
    
    return relevant_ids

def convert_to_loc(traj_centroids):
    img_height, img_width = 256, 256, # width height of image 
    # each centroid is (x,y) s.t. x is up/down direction and y is left/right direction

    traj_centroid_locs = {}
    for step, centroid in traj_centroids.items():
        if centroid == "None,None" or centroid is None:
            traj_centroid_locs[step] = ""
            continue

        centroid = centroid.split(",")
        x_orig,y_orig = float(centroid[0]), float(centroid[1])

        x = str(int(x_orig/1024 * 256))
        y = str(int(y_orig/1024 * 256))

        # format x to string such that we left-pad with up to four 0's
        x, y = x.zfill(4), y.zfill(4)

        traj_centroid_locs[step] = f"<loc{x}><loc{y}>"

    return traj_centroid_locs

def masks_path_to_seg_tokens(masks_path):
    mask_bytes_list = np.load(masks_path, allow_pickle=True)['mask_bytes']
    masks = [np.array(Image.open(io.BytesIO(mask_bytes))) for mask_bytes in mask_bytes_list]

    # resize and convert masks to paligemma-compatible tensors
    tensor_masks = []
    bbox_strs = []

    for step, mask in enumerate(masks):
        seg_idxs = np.argwhere(mask)

        # if the seg mask is null, add a seg mask of zeros to the batch for now. we'll filter this mask out using the empty bbox str later
        if len(seg_idxs) == 0:
            bbox_strs.append("")
            tensor_masks.append(tf.zeros((1, 64, 64, 1), dtype=tf.float32))
            continue

        # x and y refer to up/down and left/right directions respectively
        x_min, x_max = tf.cast(np.min(seg_idxs[:, 0]), tf.int32), tf.cast(np.max(seg_idxs[:, 0]), tf.int32)
        y_min, y_max = tf.cast(np.min(seg_idxs[:, 1]), tf.int32), tf.cast(np.max(seg_idxs[:, 1]), tf.int32)

        mask = tf.convert_to_tensor(mask, dtype=tf.uint8)
        mask = tf.image.resize(
            mask[None, x_min:x_max+1, y_min:y_max+1, None],
            [64, 64],
            method='bilinear',
            antialias=True,
        )

        tensor_masks.append(mask)

        loc = lambda c: (str(int(c/1024 * 256))).zfill(4)
        lx_min, lx_max, ly_min, ly_max = loc(x_min), loc(x_max), loc(y_min), loc(y_max)
        bbox_strs.append(f'<loc{lx_min}><loc{ly_min}><loc{lx_max}><loc{ly_max}>')

    tensor_masks = tf.concat(tensor_masks, axis=0)
    all_mask_tokens =  encode_to_codebook_indices(VQVAE_CKPT, tensor_masks).numpy()

    seg_tokens = {}
    for step, mask_tokens in enumerate(all_mask_tokens):
        if bbox_strs[step] == "": # filter out null masks
            seg_tokens[step] = ""
        else:
            mask_tokens_str = "".join([f"<seg{str(token).zfill(3)}>" for token in mask_tokens])
            seg_tokens[step] = bbox_strs[step] + mask_tokens_str
    
    return seg_tokens

def process_traj(info):

    if not info:
        return {} # don't process empty trajectory

    filtered_info = info.copy()

     # 1. remove irrelevant ids
    relevant_ids = remove_irrelevant_objs(info['obj_id_to_name'], info['language_label'])
    filtered_info['obj_id_to_name'] = {obj_id: info['obj_id_to_name'][obj_id] for obj_id in relevant_ids}
    filtered_info['obj_id_to_centroids'] = {obj_id: info['obj_id_to_centroids'][obj_id] for obj_id in relevant_ids}

    # 2. convert obj centroids to <loc> format
    new_info = {}
    new_info['obj_centroids'] = {}
    for obj_id, obj_centroids in filtered_info['obj_id_to_centroids'].items():
        new_info['obj_centroids'][obj_id] = convert_to_loc(obj_centroids)
    
    # 3. convert gripper centroids to <loc> format
    new_info['gripper_centroids'] = convert_to_loc(info['gripper_centroids'])

    # 4. convert masks to <seg> tokens
    new_info['obj_masks'] = {}
    for obj_id in filtered_info['obj_id_to_name']:
        traj_masks_path = f"{filtered_info['mask_path']}/{obj_id}.npz"
        new_info['obj_masks'][obj_id] = masks_path_to_seg_tokens(masks_path=traj_masks_path)
    
    # add back remaining keys
    new_info['language_label'] = info['language_label']

    return new_info

def main(dataset):
    json_path = f'/global/scratch/users/riadoshi/vla/generated_data/{dataset}/jsons/all'
    final_json_path = f'/global/scratch/users/riadoshi/vla/generated_data/{dataset}/jsons/{dataset}_reasonings.json'


    final_dict = {}
    if os.path.exists(final_json_path):
        print("loading existing JSON!", flush=True)
        with open(final_json_path, "r") as file:
            final_dict = json.load(file)

    write_interval = 500

    c = 0
    
    for file in sorted(os.listdir(json_path)):
        print(f"LOADING {file} **********************************", flush=True)
        with open(f'{json_path}/{file}', 'r') as f:
            batch_data = json.load(f)

        for traj_idx, traj_info in batch_data.items():
            if traj_idx in final_dict:
                print(f"SKIPPING TRAJ {traj_idx} from file {file} *************************", flush=True)
                continue

            print(f"PROCESSING TRAJ {traj_idx} from file {file}****************************", flush=True)
            new_info = process_traj(traj_info)
            final_dict[traj_idx] = new_info
            
            if c%write_interval==0:
                with open(final_json_path, 'w') as f:
                    json.dump(final_dict, f)
            c+=1

    with open(final_json_path, 'w') as f:
        json.dump(final_dict, f)
    print(f"finished processing {dataset}!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    main(args.dataset)



