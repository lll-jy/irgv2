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
EVAL_CONFIG=eval_conf
EVAL_OUT_DIR=evaluation
DOC_PORT=8080
PYTHON=python3
DB_NAME=rtd
META_DIR=meta
SRC_DATA_DIR=src
LOG_LEVEL=INFO

install:
	${PYTHON} -m pip install --upgrade pip
	${PYTHON} -m pip install -r requirements.txt
	${PYTHON} setup.py install
	${PYTHON} -m pip install -e .

update:
	${PYTHON} setup.py install
	${PYTHON} -m pip install -e .

docs:
	pdoc --http localhost:${DOC_PORT} -c latex_math=True irg docs examples

prepare:
	${PYTHON} process.py database ${DB_NAME} \
		--src_data_dir ${SRC_DATA_DIR} \
		--data_dir ${DATA_DIR} \
        --meta_dir ${META_DIR} \
        --out ${DB_CONFIG} \
        --redo_meta \
        --redo_data

train_gpu:
	${PYTHON} -m torch.distributed.launch \
		--nproc_per_node=${NUM_GPUS} \
		--master_port=${PORT} \
		main.py --log_level ${LOG_LEVEL} train_gen \
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
	${PYTHON} -W ignore main.py --log_level ${LOG_LEVEL} train_gen \
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
	${PYTHON} -m torch.distributed.launch \
		--nproc_per_node=${NUM_GPUS} \
		--master_port=${PORT} \
		main.py --log_level ${LOG_LEVEL} train_gen \
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
	${PYTHON} main.py --log_level ${LOG_LEVEL} train_gen \
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

evaluate:
	mkdir -p ${EVAL_OUT_DIR}
	mkdir -p ${EVAL_OUT_DIR}/tables
	${PYTHON} main.py --log_level ${LOG_LEVEL} evaluate \
		--real_db_dir ${CACHE_DB} \
		--fake_db_dir ${CACHE_DB_FAKE} \
		--evaluator_path ${EVAL_CONFIG}/constructor.json \
		--evaluate_path ${EVAL_CONFIG}/evaluate.json \
		--save_eval_res_to ${EVAL_OUT_DIR}/trivial \
		--save_complete_result_to ${EVAL_OUT_DIR}/complete \
		--save_synthetic_tables_to ${EVAL_OUT_DIR}/tables/synthetic \
		--save_tables_to ${EVAL_OUT_DIR}/tables/real \
		--save_visualization_to ${EVAL_OUT_DIR}/visualization \
		--save_all_res_to ${EVAL_OUT_DIR}/result
