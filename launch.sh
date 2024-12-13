export TPU_VISIBLE_DEVICES=0,1,2,3
export TPU_CHIPS_PER_HOST_BOUNDS=1,4,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_MESH_CONTROLLER_ADDRESS=localhost:8476
export TPU_MESH_CONTROLLER_PORT=8476
export WANDB_DISABLED=True

export GOOGLE_APPLICATION_CREDENTIALS="/nfs/nfs3/mitsuhiko/codes/rail-tpus-e97862aa96cf.json"

python test_podv5_cmu.py