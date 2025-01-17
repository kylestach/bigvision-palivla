import numpy as np
import argparse
import os
import json


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

        x = str(int(x_orig/256 * 1024))
        y = str(int(y_orig/256 * 1024))

        # format x to string such that we left-pad with up to four 0's
        x, y = x.zfill(4), y.zfill(4)

        traj_centroid_locs[step] = f"<loc{x}><loc{y}>"

    return traj_centroid_locs


def process_traj(info):

    if not info:
        return {} # don't process empty trajectory

    # 2. convert hand centroids to <loc> format
    new_info = {}
    new_info['right_hand_centroids'] = convert_to_loc(info["right_hand_centroids"])
    new_info['language_label'] = info['language_label']

    return new_info


def main(dataset):
    json_path = f'/home/oiermees/generated_ecot/{dataset}/jsons/'
    final_json_path = f'/home/oiermees/generated_ecot/{dataset}/jsons/{dataset}_reasonings.json'

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

            if c % write_interval == 0:
                with open(final_json_path, 'w') as f:
                    json.dump(final_dict, f)
            c += 1

    with open(final_json_path, 'w') as f:
        json.dump(final_dict, f)
    print(f"finished processing {dataset}!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    main(args.dataset)