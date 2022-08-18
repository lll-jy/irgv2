PORT=1234
MTYPE=affecting
SCALING=1
EVAL_CONFIG=eval_conf
DOC_PORT=8080
PYTHON=python3
DB_NAME=rtd
SRC_DATA_DIR=src
LOG_LEVEL=INFO
TEMP_CACHE=.temp.nosync
FAKE_DB=

DATA_OUTPUT_DIR=data
MODEL_OUTPUT_DIR=output
GENERATE_OUTPUT_DIR=generated
EVAL_OUTPUT_DIR=evaluation

EXTRACT_TO=IRGv2_copy

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
	mkdir -p ${DATA_OUTPUT_DIR}
	${PYTHON} process.py database ${DB_NAME} \
		--src_data_dir ${SRC_DATA_DIR} \
		--data_dir ${DATA_OUTPUT_DIR}/data \
        --meta_dir ${DATA_OUTPUT_DIR}/metadata \
        --out ${DATA_OUTPUT_DIR}/db_config.json \
        --redo_meta \
        --redo_data

train_gpu:
	mkrit -p ${MODEL_OUTPUT_DIR}
	${PYTHON} -W ignore -m torch.distributed.launch \
		--nproc_per_node=${NUM_GPUS} \
		--master_port=${PORT} \
		main.py --log_level ${LOG_LEVEL} \
			--temp_cache ${TEMP_CACHE} \
			train_gen \
			--distributed \
			--db_config_path ${DATA_OUTPUT_DIR}/db_config.json \
			--data_dir ${DATA_OUTPUT_DIR}/data \
			--mtype ${MTYPE} \
			--db_dir_path ${MODEL_OUTPUT_DIR}/real_db \
			--aug_resume \
			--default_tab_trainer_distributed True \
			--default_deg_trainer_distributed True \
			--default_tab_trainer_autocast True \
			--default_deg_trainer_autocast True \
			--default_tab_trainer_log_dir ${MODEL_OUTPUT_DIR}/tf/tab \
			--default_deg_trainer_log_dir ${MODEL_OUTPUT_DIR}/tf/deg \
			--default_tab_trainer_ckpt_dir ${MODEL_OUTPUT_DIR}/ckpt/tab \
			--default_deg_trainer_ckpt_dir ${MODEL_OUTPUT_DIR}/ckpt/deg \
			--skip_generate

train_cpu:
	mkdir -p ${MODEL_OUTPUT_DIR}
	${PYTHON} -W ignore main.py --log_level ${LOG_LEVEL} train_gen \
		--db_config_path ${DATA_OUTPUT_DIR}/db_config.json \
		--data_dir ${DATA_OUTPUT_DIR}/data \
		--mtype ${MTYPE} \
		--db_dir_path ${MODEL_OUTPUT_DIR}/real_db \
		--aug_resume \
		--default_tab_trainer_log_dir ${MODEL_OUTPUT_DIR}/tf/tab \
		--default_deg_trainer_log_dir ${MODEL_OUTPUT_DIR}/tf/deg \
		--default_tab_trainer_ckpt_dir ${MODEL_OUTPUT_DIR}/ckpt/tab \
		--default_deg_trainer_ckpt_dir ${MODEL_OUTPUT_DIR}/ckpt/deg \
		--skip_generate

generate_gpu:
	mkdir -p ${GENERATE_OUTPUT_DIR}
	${PYTHON} -m torch.distributed.launch \
		--nproc_per_node=${NUM_GPUS} \
		--master_port=${PORT} \
		main.py --log_level ${LOG_LEVEL} train_gen \
			--distrubted \
			--db_config_path ${DATA_OUTPUT_DIR}/db_config.json \
			--data_dir ${DATA_OUTPUT_DIR}/data \
			--mtype ${MTYPE} \
			--db_dir_path ${MODEL_OUTPUT_DIR}/real_db \
			--aug_resume \
			--skip_train \
			--default_tab_train_resume \
			--default_deg_train_resume \
			--default_tab_trainer_distributed \
			--default_deg_trainer_distributed \
			--default_tab_trainer_autocast \
			--default_deg_trainer_autocast \
			--default_tab_trainer_log_dir ${MODEL_OUTPUT_DIR}/tf/tab \
			--default_deg_trainer_log_dir ${MODEL_OUTPUT_DIR}/tf/deg \
			--default_tab_trainer_ckpt_dir ${MODEL_OUTPUT_DIR}/ckpt/tab \
			--default_deg_trainer_ckpt_dir ${MODEL_OUTPUT_DIR}/ckpt/deg \
			--save_generated_to ${GENERATE_OUTPUT_DIR}/generated \
			--default_scaling ${SCALING} \
			--save_synth_db ${GENERATE_OUTPUT_DIR}/fake_db

generate_cpu:
	mkdir -p ${GENERATE_OUTPUT_DIR}
	${PYTHON} -W ignore main.py --log_level ${LOG_LEVEL} train_gen \
		--distrubted \
		--db_config_path ${DATA_OUTPUT_DIR}/db_config.json \
		--data_dir ${DATA_OUTPUT_DIR}/data \
		--mtype ${MTYPE} \
		--db_dir_path ${MODEL_OUTPUT_DIR}/real_db \
		--aug_resume \
		--skip_train \
		--default_tab_train_resume \
		--default_deg_train_resume \
		--default_tab_trainer_log_dir ${MODEL_OUTPUT_DIR}/tf/tab \
		--default_deg_trainer_log_dir ${MODEL_OUTPUT_DIR}/tf/deg \
		--default_tab_trainer_ckpt_dir ${MODEL_OUTPUT_DIR}/ckpt/tab \
		--default_deg_trainer_ckpt_dir ${MODEL_OUTPUT_DIR}/ckpt/deg \
		--save_generated_to ${GENERATE_OUTPUT_DIR}/generated \
		--default_scaling ${SCALING} \
		--save_synth_db ${GENERATE_OUTPUT_DIR}/fake_db

evaluate:
	mkdir -p ${EVAL_OUTPUT_DIR}
	mkdir -p ${EVAL_OUTPUT_DIR}/tables
	${PYTHON} -W ignore main.py --log_level ${LOG_LEVEL} evaluate \
		--real_db_dir ${MODEL_OUTPUT_DIR}/real_db \
		--fake_db_dir ${FAKE_DB} \
		--evaluator_path ${EVAL_CONFIG}/constructor.json \
		--evaluate_path ${EVAL_CONFIG}/evaluate.json \
		--save_eval_res_to ${EVAL_OUTPUT_DIR}/trivial \
		--save_complete_result_to ${EVAL_OUTPUT_DIR}/complete \
		--save_synthetic_tables_to ${EVAL_OUTPUT_DIR}/tables/synthetic \
		--save_tables_to ${EVAL_OUTPUT_DIR}/tables/real \
		--save_visualization_to ${EVAL_OUTPUT_DIR}/visualization \
		--save_all_res_to ${EVAL_OUTPUT_DIR}/result

stop:
	pkill -9 -f main.py
	pkill -9 -f multiprocessing.fork
	pkill -9 -f multiprocessing.spawn

extract:
	rm -rf ../${EXTRACT_TO}
	mkdir ../${EXTRACT_TO}
	cp *.py ../${EXTRACT_TO}/
	cp Makefile ../${EXTRACT_TO}/
	cp requirements.txt ../${EXTRACT_TO}/
	cp -r irg ../${EXTRACT_TO}/
	cp -r examples ../${EXTRACT_TO}/
