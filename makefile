GPUS=4
OMP_NUM_THREADS=12

null:
	@echo 'nothing happend'

train_ddp:
	@OMP_NUM_THREADS=$(OMP_NUM_THREADS) torchrun --nproc_per_node=$(GPUS) train.py \
	-- --yaml_config_path=config/base_config.yaml

train_single:
	@python train.py --yaml_config_path=config/base_config.yaml