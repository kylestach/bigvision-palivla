sudo apt -y update && sudo apt -y install nfs-common
sudo mkdir -p -m 777 /nfs/nfs2
sudo mount -o rw,intr 10.30.175.26:/nfs2 /nfs/nfs2

sudo pkill -u kstachowicz
sudo usermod -u 3210 kstachowicz
sudo groupmod -g 3210 kstachowicz
sudo chown -R kstachowicz:kstachowicz /home/$USER

sudo -i -u kstachowicz bash << EOF

git config --global --add safe.directory '*'

rm -f .bashrc
mkdir -p .cache
ln -s /nfs/nfs2/users/kstachowicz/.bashrc .bashrc
ln -s /nfs/nfs2/users/kstachowicz/.netrc .netrc
ln -s /nfs/nfs2/users/kstachowicz/.env .env
ln -s /nfs/nfs2/users/kstachowicz/.cache/huggingface .cache/huggingface
EOF
