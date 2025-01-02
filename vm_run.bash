uv run --prerelease=allow python palivla/train.py --config palivla/configs/bridge_config.py \
	--config.eval_interval 10 \
	--config.log_interval 10 \
	--config.save_interval 1000 \
	--config.data_axis_size 1 \
	--config.fsdp_axis_size -1 \
	--config.paligemma_weights_path models/paligemma-3b-pt-224.f16.npz \
	--config.language_tokenizer_path models/paligemma_tokenizer.model \
	--config.run_name overfit \
