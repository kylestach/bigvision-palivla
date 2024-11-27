export TPU_VISIBLE_DEVICES=1
export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 
export TPU_HOST_BOUNDS=1,1,1 
export TPU_MESH_CONTROLLER_ADDRESS=localhost:8476 
export TPU_MESH_CONTROLLER_PORT=8476
python inference_action_value.py \
--config.save_interval 50000 \
--config.data_axis_size 1 \
--config.fsdp_axis_size -1 \
--config.paligemma_weights_path models/paligemma-3b-pt-224.f16.npz \
--config.language_tokenizer_path models/paligemma_tokenizer.model \
--config.batch_size 64 \
--config.save_path gs://rail-tpus-mitsuhiko-central2/logs/test \
--config.dataset_kwargs.oxe_kwargs.data_dir gs://rail-orca-central2/resize_256_256