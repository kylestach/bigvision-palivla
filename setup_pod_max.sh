sudo apt -y update && sudo apt -y install nfs-common
sudo mkdir -p -m 777 /nfs/aidm_nfs
sudo mount -o rw,intr 10.43.108.242:/aidm_nfs_europe /nfs/aidm_nfs

sudo usermod -u 3210 maxsobolmark
sudo groupmod -g 3210 maxsobolmark
sudo chown -R maxsobolmark:maxsobolmark /home/maxsobolmark

sudo -i -u maxsobolmark bash << EOF

git config --global --add safe.directory '*'

rm -f .bashrc
ln -s /nfs/aidm_nfs/max/.bashrc .bashrc
ln -s /nfs/aidm_nfs/max/.netrc .netrc
EOF

/nfs/aidm_nfs/max/miniconda3/bin/conda init bash
