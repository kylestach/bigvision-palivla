## Data Generation Pipeline 

The general pipeline is to  <br>
1. Run `generate_object_names.py`. This uses Molmo (small version) to save down relevant object to `jsons/object_names/batchi.json` <br>
2. Run `generate_everything_else.py`. This queries DETR for gripper centroids, Molmo MoE for object coordinates, and SAM to propagate these object coordinates through a trajectory to generate segmentation masks. This will save down gripper centroids, object centroids, and object seg masks to `jsons/all/batchi.json`. Make sure that the `TRAJS_TO_PROCESS` key is consistent across steps (1) and (2)! Which json file to read from is determined accordingly. 
3. Run `post_hoc.py`. This will perform four post-processing steps: 
- remove objects that are irrelevant to the task using a simple heuristic
- convert all object centroids to the `<loc>` format that PaliGemma expects
- convert all gripper centroids to the `<loc>` format that PaliGemma expects
- runs PaliGemma's VQ tokenizer on the object segmentation masks, to get the `<loc><seg>` series of tokens 

