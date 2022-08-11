PORT=1234
DB_CONFIG=config.json
DATA_DIR=.
MTYPE=affecting
CACHE_DB=real_db
OUT_DIR=generated
LOG_DIR=logs
CKPT_DIR=checkpoints
SCALING=1
CACHE_DB_FAKE=fake_db

install:
	pip3 install -r requirements.txt
	python3 setup.py install

docs:
	pdoc --http localhost:8080 -c latex_math=True irg docs

train_gpu:
	python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${PORT} main.py \
		--distrubted \
		--db_config_path ${DB_CONFIG} \
		--data_dir ${DATA_DIR} \
		--mtype ${MTYPE} \
		--db_dir_path ${CACHE_DB} \
		--aug_resume \
		--default_tab_trainer_distributed \
		--default_deg_trainer_distributed \
		--default_tab_trainer_autocast \
		--default_deg_trainer_autocast \
		--default_tab_trainer_log_dir ${LOG_DIR}/tab \
		--default_deg_trainer_log_dir ${LOG_DIR}/deg \
		--default_tab_trainer_ckpt_dir ${CKPT_DIR}/tab \
		--default_deg_trainer_ckpt_dir ${CKPT_DIR}/deg \
		--skip_generate

train_cpu:
	python3 main.py \
		--db_config_path ${DB_CONFIG} \
		--data_dir ${DATA_DIR} \
		--mtype ${MTYPE} \
		--db_dir_path ${CACHE_DB} \
		--aug_resume \
		--default_tab_trainer_log_dir ${LOG_DIR}/tab \
		--default_deg_trainer_log_dir ${LOG_DIR}/deg \
		--default_tab_trainer_ckpt_dir ${CKPT_DIR}/tab \
		--default_deg_trainer_ckpt_dir ${CKPT_DIR}/deg \
		--skip_generate

generate_gpu:
	python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${PORT} main.py \
		--distrubted \
		--db_config_path ${DB_CONFIG} \
		--data_dir ${DATA_DIR} \
		--mtype ${MTYPE} \
		--db_dir_path ${CACHE_DB} \
		--aug_resume \
		--skip_train \
		--default_tab_train_resume \
		--default_deg_train_resume \
		--default_tab_trainer_distributed \
		--default_deg_trainer_distributed \
		--default_tab_trainer_autocast \
		--default_deg_trainer_autocast \
		--default_tab_trainer_log_dir ${LOG_DIR}/tab \
		--default_deg_trainer_log_dir ${LOG_DIR}/deg \
		--default_tab_trainer_ckpt_dir ${CKPT_DIR}/tab \
		--default_deg_trainer_ckpt_dir ${CKPT_DIR}/deg \
		--save_generated_to ${OUT_DIR} \
		--default_scaling ${SCALING} \
		--save_synth_db ${CACHE_DB_FAKE}

generate_cpu:
	python3 main.py \
		--distrubted \
		--db_config_path ${DB_CONFIG} \
		--data_dir ${DATA_DIR} \
		--mtype ${MTYPE} \
		--db_dir_path ${CACHE_DB} \
		--aug_resume \
		--skip_train \
		--default_tab_train_resume \
		--default_deg_train_resume \
		--default_tab_trainer_log_dir ${LOG_DIR}/tab \
		--default_deg_trainer_log_dir ${LOG_DIR}/deg \
		--default_tab_trainer_ckpt_dir ${CKPT_DIR}/tab \
		--default_deg_trainer_ckpt_dir ${CKPT_DIR}/deg \
		--save_generated_to ${OUT_DIR} \
		--default_scaling ${SCALING} \
		--save_synth_db ${CACHE_DB_FAKE}
