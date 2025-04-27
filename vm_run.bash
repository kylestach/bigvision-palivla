uv run --prerelease=allow python scripts/train.py --config /nfs/nfs2/users/oiermees/bigvision-palivla/configs/actions_only/bridge_finetune_new.py \
	--config.eval_interval 10 \
	--config.cot_path gs://multi-robot-bucket2/data/generated_data \
	--config.save_interval 1000 \
	--config.batch_size 256 \
	--config.viz_interval 25 \
	--config.eval_batch_size 8 \
	--config.save_path=gs://multi-robot-bucket2/runs/vla \
    --config.dataset_kwargs.oxe_kwargs.data_dir=gs://rail-orca-central2/resize_256_256 \


# --config.cot_path gs://multi-robot-bucket2/data/generated_data \

	
