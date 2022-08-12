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

install:
	pip3 install -r requirements.txt
	python3 setup.py install

docs:
	pdoc --http localhost:${DOC_PORT} -c latex_math=True irg docs

train_gpu:
	python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${PORT} main.py train_gen \
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
	python3 main.py train_gen \
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
	python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${PORT} main.py train_gen \
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
	python3 main.py train_gen \
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
	python3 main.py evaluate \
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
