uv run --prerelease=allow python scripts/train.py --config configs/cot_cross_emb_config.py \
	--config.eval_interval 100 \
	--config.save_interval 1000 \
	--config.batch_size 32 \
	--config.viz_interval 100 \
	--config.eval_batch_size 8 \
	--config.cot_path gs://multi-robot-bucket2/data/generated_data \
	--config.save_path=gs://multi-robot-bucket2/runs/vla \
    --config.dataset_kwargs.oxe_kwargs.data_dir=gs://rail-orca-central2/resize_256_256 \

