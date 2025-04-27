#!/bin/bash

# Activate ssh key 
mkdir -p /home/noam/.ssh
mv id_ed25519 /home/noam/.ssh
eval "$(ssh-agent -s)"
ssh-add id_ed25519
touch /home/noam/.ssh/known_hosts
ssh-keyscan github.com >> /home/noam/.ssh/known_hosts
sudo tar -xvzf ngrok-v3-stable-linux-amd64.tgz -C /usr/local/bin
# Need to set token for ngrok
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
git clone git@github.com:catglossop/bigvision-palivla.git --recursive
cd ~/bigvision-palivla
git pull
git submodule sync --recursive
cd ~/bigvision-palivla/octo
git fetch 
git checkout origin/main
git branch main -f 
git checkout main
cd ~/bigvision-palivla
source .venv/bin/activate
uv python pin 3.11.12
uv venv --python=python3.11.12
uv sync --extra tpu  

# For inference
uv pip install opencv-python
sudo apt-get install libgl1 -y
uv pip install Flask
uv pip install flask-ngrok
uv pip install ngrok
uv pip install google-cloud-logging
uv pip install google-cloud-storage
sudo cp /tmp/ngrok/ngrok /usr/local/bin
# Modify the flask_ngrok file to use path /usr/bin

# for inference - need to move ngrok from /tmp/tmp/ngrok to /usr