#!/bin/bash

# Check if a TPU VM name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <tpu-vm-name>"
    exit 1
fi

TPU_VM_NAME=$1
INIT=$2
UPDATE=$3
PROJECT="vlm-guidance"
ZONE="us-central2-b"
API_KEY=$4
CONFIG_NAME=$5

echo "Update? $UPDATE"
echo "Init? $INIT"

# Cache file for TPU name/zone mapping
CACHE_FILE="$HOME/.cache/tpus"
mkdir -p "$(dirname "$CACHE_FILE")"

TPU_INFO=$(gcloud compute tpus tpu-vm describe $TPU_VM_NAME --project=$PROJECT --zone=$ZONE --format=json 2>/dev/null)
IP = $(echo "$TPU_INFO" | jq '.networkEndpoints[0].ipAddress')
echo "TPU_INFO: $TPU_INFO"

echo "TPU_VM_NAME: $TPU_VM_NAME"
echo "ZONE: $ZONE"
echo "Number of workers: $NUM_WORKERS"

# Copy the source directory to the TPU VM
if $INIT; then
    echo "Copying source directory to TPU VM"
    # gcloud alpha compute tpus tpu-vm scp ~/bigvision-palivla/ $TPU_VM_NAME: --recurse --zone=us-central2-b
    gcloud alpha compute tpus tpu-vm scp ~/.ssh/id_ed25519 $TPU_VM_NAME: --zone=us-central2-b
    gcloud alpha compute tpus tpu-vm scp tpu_config_cat.sh $TPU_VM_NAME: --zone=us-central2-b
    gcloud alpha compute tpus tpu-vm scp update.sh $TPU_VM_NAME: --zone=us-central2-b
    gcloud alpha compute tpus tpu-vm scp ~/ngrok-v3-stable-linux-amd64.tgz $TPU_VM_NAME: --zone=us-central2-b
 
    echo "Initializing TPU VM"
    gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME --zone=us-central2-b --command="bash tpu_config_cat.sh"
fi 

if $UPDATE; then
    echo "Updating source directory on TPU VM"
    git add *
    git commit -m "Update"
    git push
    cd octo
    git add *
    git commit -m "Update"
    git push
    cd ..
    gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME --zone=us-central2-b --command="bash update.sh"
fi
echo "Launching Script"
gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME --zone=us-central2-b --command="cd ~/bigvision-palivla &&
                                                                                  source ~/.local/bin/env &&
                                                                                  source .venv/bin/activate &&
                                                                                  uv run wandb login $API_KEY &&
                                                                                  python scripts/train.py --config configs/$CONFIG_NAME"