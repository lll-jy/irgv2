DB_NAME=alset
EXP_NAME=small
DATA_VERSION=samples
TAB_TRAINER_CFG=default
DEG_TRAINER_CFG=default
SER_TRAINER_CFG=default
TAB_TRAIN_CFG=default
DEG_TRAIN_CFG=default
SER_TRAIN_CFG=default
SCALING=1
EVALUATOR_CFG=default
EVALUATE_CFG=default
USE_SAMPLE=/samples
DB_VERSION=small
MTYPE=affecting
BASE_DIR=examples
OUT_SUFFIX=.nosync
SINGLE_NAME=adults
GENERATE_VERSION=
SP_SCALE=

prepare_small_alset:
	python3.9 process.py database alset \
        --src_data_dir examples/data${OUT_SUFFIX}/alset \
        --data_dir ${BASE_DIR}/data${OUT_SUFFIX}/alset/samples \
        --meta_dir examples/alset/metadata/results \
        --out ${BASE_DIR}/data${OUT_SUFFIX}/alset/samples_db_config.json \
        --redo_meta --redo_data \
        --tables \
            personal_data \
            sis_academic_career \
            sis_academic_program_offer \
            sis_academic_program \
            sis_plan_offer \
            sis_academic_plan \
            sis_enrolment \
        --sample 50


prepare_model_compare:
	python3.9 process.py database alset \
		--src_data_dir examples/data${OUT_SUFFIX}/alset \
		--data_dir ${BASE_DIR}/data${OUT_SUFFIX}/alset/model_compare \
		--meta_dir examples/alset/metadata/results \
		--out ${BASE_DIR}/data${OUT_SUFFIX}/alset/model_compare_db_config.json \
		--redo_meta --redo_data \
		--tables \
			personal_data \
			sis_milestone \
			module_offer \
			module_enrolment \
			uci_gym \
		--sample 5000


prepare_wifi:
	python3.9 process.py database alset \
		--src_data_dir examples/data${OUT_SUFFIX}/alset \
		--data_dir ${BASE_DIR}/data${OUT_SUFFIX}/alset/gen_wifi \
		--meta_dir examples/alset/metadata/results \
		--out ${BASE_DIR}/data${OUT_SUFFIX}/alset/gen_wifi_db_config.json \
		--redo_meta --redo_data \
		--tables \
			personal_data \
			wifi


prepare_all_alset:
	echo TODO


prepare_table_alset:
	python3.9 process.py database alset \
        --src_data_dir examples/data${OUT_SUFFIX}/alset \
        --data_dir ${BASE_DIR}/data${OUT_SUFFIX}/alset/table \
        --meta_dir examples/alset/metadata/results \
        --out ${BASE_DIR}/data${OUT_SUFFIX}/alset/table_db_config.json \
        --redo_meta --redo_data \
        --tables \
            personal_data \
        --sample 50


prepare_airbnb:
	python3.9 process.py database airbnb \
		--src_data_dir examples/data${OUT_SUFFIX}/airbnb \
		--data_dir ${BASE_DIR}/data${OUT_SUFFIX}/airbnb/processed \
		--meta_dir examples/airbnb/metadata/results \
		--out ${BASE_DIR}/data${OUT_SUFFIX}/airbnb/processed_db_config.json \
		--redo_meta --redo_data \
		--sample 10000

prepare_single:
	python3.9 process.py database ${SINGLE_NAME} \
		--src_data_dir examples/data${OUT_SUFFIX}/${SINGLE_NAME} \
		--data_dir ${BASE_DIR}/data${OUT_SUFFIX}/${SINGLE_NAME}/${SINGLE_NAME} \
		--meta_dir examples/${SINGLE_NAME}/metadata \
		--out ${BASE_DIR}/data${OUT_SUFFIX}/${SINGLE_NAME}/${SINGLE_NAME}_db_config.json

train:
	-python3.9 -W ignore main.py --log_level WARN --num_processes 10 --temp_cache .temp${OUT_SUFFIX} train_gen \
        --db_config_path ${BASE_DIR}/data${OUT_SUFFIX}/${DB_NAME}/${DATA_VERSION}_db_config.json \
        --data_dir ${BASE_DIR}/data${OUT_SUFFIX}/${DB_NAME}/${DATA_VERSION}${USE_SAMPLE} \
        --db_dir_path ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${DB_VERSION}/real_db \
        --aug_resume \
        --default_tab_trainer_log_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/tf/tab \
        --default_deg_trainer_log_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/tf/deg \
        --default_ser_trainer_log_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/tf/tab \
        --default_tab_trainer_ckpt_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/ckpt/tab \
        --default_deg_trainer_ckpt_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/ckpt/deg \
        --default_ser_trainer_ckpt_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/ckpt/tab \
        --skip_generate >> log.txt
	du -sh .temp${OUT_SUFFIX}
	du -sh ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}


train_cfg:
	-python3.9 -W ignore main.py --log_level WARN --num_processes 10 --temp_cache .temp${OUT_SUFFIX} train_gen \
        --db_config_path ${BASE_DIR}/data${OUT_SUFFIX}/${DB_NAME}/${DATA_VERSION}_db_config.json \
        --data_dir ${BASE_DIR}/data${OUT_SUFFIX}/${DB_NAME}/${DATA_VERSION}${USE_SAMPLE} \
        --db_dir_path ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${DB_VERSION}/real_db \
        --mtype ${MTYPE} \
        --aug_resume \
        --default_tab_trainer_args config/trainer/${TAB_TRAINER_CFG}.json \
        --default_deg_trainer_args config/trainer/${DEG_TRAINER_CFG}.json \
        --default_ser_trainer_args config/trainer/${SER_TRAINER_CFG}.json \
        --default_tab_train_args config/train/${TAB_TRAIN_CFG}.json \
        --default_deg_train_args config/train/${DEG_TRAIN_CFG}.json \
        --default_ser_train_args config/train/${SER_TRAIN_CFG}.json \
        --default_tab_trainer_log_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/tf/tab \
        --default_deg_trainer_log_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/tf/deg \
        --default_ser_trainer_log_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/tf/tab \
        --default_tab_trainer_ckpt_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/ckpt/tab \
        --default_deg_trainer_ckpt_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/ckpt/deg \
        --default_ser_trainer_ckpt_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/ckpt/ser \
        --skip_generate >> log.txt
	du -sh .temp${OUT_SUFFIX}
	du -sh ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}


inspect_sizes:
	echo data:
	-du -sh ${BASE_DIR}/data${OUT_SUFFIX}/${DB_NAME}/${DATA_VERSION}${USE_SAMPLE}
	-du -sh ${BASE_DIR}/generated${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}${GENERATE_VERSION}/generated
	echo cache:
	-du -sh .temp${OUT_SUFFIX}
	echo checkpoints:
	-du -sh ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/ckpt/tab
	-du -sh ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/ckpt/deg
	echo tensorboard:
	-du -sh ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/tf/tab
	-du -sh ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/tf/deg
	echo database:
	-du -sh ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${DB_VERSION}/real_db
	-du -sh ${BASE_DIR}/generated${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}${GENERATE_VERSION}/fake_db


generate:
	-mkdir -p ${BASE_DIR}/generated${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}${GENERATE_VERSION}
	-python3.9 -W ignore main.py --log_level WARN --temp_cache .temp${OUT_SUFFIX} --num_processes 10 train_gen \
        --db_config_path ${BASE_DIR}/data${OUT_SUFFIX}/${DB_NAME}/${DATA_VERSION}_db_config.json \
        --data_dir ${BASE_DIR}/data${OUT_SUFFIX}/${DB_NAME}/${DATA_VERSION}${USE_SAMPLE} \
        --db_dir_path ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${DB_VERSION}/real_db \
        --aug_resume \
        --skip_train \
        --default_tab_train_resume True \
        --default_deg_train_resume True \
        --default_tab_trainer_log_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/tf/tab \
        --default_deg_trainer_log_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/tf/deg \
        --default_ser_trainer_log_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/tf/tab \
        --default_tab_trainer_ckpt_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/ckpt/tab \
        --default_deg_trainer_ckpt_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/ckpt/deg \
        --default_ser_trainer_ckpt_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/ckpt/ser \
        --save_generated_to ${BASE_DIR}/generated${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}${GENERATE_VERSION}/generated \
        --save_synth_db ${BASE_DIR}/generated${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}${GENERATE_VERSION}/fake_db >> log.txt
	du -sh .temp${OUT_SUFFIX}
	du -sh ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}
	du -sh ${BASE_DIR}/generated${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}${GENERATE_VERSION}


generate_cfg:
	-mkdir -p ${BASE_DIR}/generated${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}${GENERATE_VERSION}
	-python3.9 -W ignore main.py --log_level WARN --temp_cache .temp${OUT_SUFFIX} --num_processes 10 train_gen \
        --db_config_path ${BASE_DIR}/data${OUT_SUFFIX}/${DB_NAME}/${DATA_VERSION}_db_config.json \
        --data_dir ${BASE_DIR}/data${OUT_SUFFIX}/${DB_NAME}/${DATA_VERSION}${USE_SAMPLE} \
        --db_dir_path ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${DB_VERSION}/real_db \
        --aug_resume \
		--default_tab_trainer_args config/trainer/${TAB_TRAINER_CFG}.json \
		--default_deg_trainer_args config/trainer/${DEG_TRAINER_CFG}.json \
		--default_ser_trainer_args config/trainer/${SER_TRAINER_CFG}.json \
		--default_tab_train_args config/train/${TAB_TRAIN_CFG}.json \
		--default_deg_train_args config/train/${DEG_TRAIN_CFG}.json \
		--default_ser_train_args config/train/${SER_TRAIN_CFG}.json \
        --skip_train \
        --default_tab_train_resume True \
        --default_deg_train_resume True \
        --default_scaling ${SCALING} \
        --scaling ${SP_SCALE} \
        --default_tab_trainer_log_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/tf/tab \
        --default_deg_trainer_log_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/tf/deg \
        --default_ser_trainer_log_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/tf/tab \
        --default_tab_trainer_ckpt_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/ckpt/tab \
        --default_deg_trainer_ckpt_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/ckpt/deg \
        --default_ser_trainer_ckpt_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/ckpt/ser \
        --save_generated_to ${BASE_DIR}/generated${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}${GENERATE_VERSION}/generated \
        --save_synth_db ${BASE_DIR}/generated${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}${GENERATE_VERSION}/fake_db >> log.txt
	du -sh .temp${OUT_SUFFIX}
	du -sh ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}
	du -sh ${BASE_DIR}/generated${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}${GENERATE_VERSION}


evaluate:
	-mkdir -p ${BASE_DIR}/evaluate${OUT_SUFFIX}
	-mkdir -p ${BASE_DIR}/evaluate${OUT_SUFFIX}/${DB_NAME}
	-mkdir -p ${BASE_DIR}/evaluate${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}
	-python3.9  -W ignore main.py --log_level WARN --temp_cache .temp${OUT_SUFFIX} --num_processes 10 evaluate \
		--real_db_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${DB_VERSION}/real_db \
		--fake_db_dir ${BASE_DIR}/generated${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}${GENERATE_VERSION}/fake_db \
		--save_eval_res_to ${BASE_DIR}/evaluate${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/results \
		--save_complete_result_to ${BASE_DIR}/evaluate${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/complete \
		--save_tables_to ${BASE_DIR}/evaluate${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/tables \
		--save_visualization_to ${BASE_DIR}/evaluate${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/visualization \
		--save_all_res_to ${BASE_DIR}/evaluate${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/result.pkl >> log.txt
	du -sh .temp${OUT_SUFFIX}
	du -sh ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}
	du -sh ${BASE_DIR}/generated${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}${GENERATE_VERSION}
	du -sh ${BASE_DIR}/evaluate${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}


evaluate_cfg:
	-mkdir -p ${BASE_DIR}/evaluate${OUT_SUFFIX}
	-mkdir -p ${BASE_DIR}/evaluate${OUT_SUFFIX}/${DB_NAME}
	-mkdir -p ${BASE_DIR}/evaluate${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}
	-python3.9  -W ignore main.py --log_level WARN --temp_cache .temp${OUT_SUFFIX} --num_processes 10 evaluate \
		--real_db_dir ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${DB_VERSION}/real_db \
		--fake_db_dir ${BASE_DIR}/generated${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}${GENERATE_VERSION}/fake_db \
		--evaluator_path config/evaluator/${EVALUATOR_CFG}.json \
		--evaluate_path config/evaluator/${EVALUATE_CFG}.json \
		--save_eval_res_to ${BASE_DIR}/evaluate${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/results \
		--save_complete_result_to ${BASE_DIR}/evaluate${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/complete \
		--save_tables_to ${BASE_DIR}/evaluate${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/tables \
		--save_visualization_to ${BASE_DIR}/evaluate${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/visualization \
		--save_all_res_to ${BASE_DIR}/evaluate${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/result.pkl >> log.txt
	du -sh .temp${OUT_SUFFIX}
	du -sh ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}
	du -sh ${BASE_DIR}/generated${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}${GENERATE_VERSION}
	du -sh ${BASE_DIR}/evaluate${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}


kill:
	-pkill -9 -f main.py
	-pkill -9 -f torch.multiprocessing.spawn
	-pkill -9 -f torch.multiprocessing.fork


clear: kill
	-rm -r .temp
	-rm -r .temp${OUT_SUFFIX}
	-rm -r ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/
	-rm -r ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${DB_VERSION}/real_db


clear_db: kill
	-rm -r ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${DB_VERSION}/real_db


clear_ckpt: kill
	-rm -r ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/ckpt
	-rm -r ${BASE_DIR}/model${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/tf


clear_gen: kill
	-rm -r ${BASE_DIR}/generated${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}${GENERATE_VERSION}/


clear_eval: kill
	-rm -r ${BASE_DIR}/evaluate${OUT_SUFFIX}/${DB_NAME}/${EXP_NAME}/


clear_all: clear_eval clear_gen clear_ckpt clear


rm_log:
	-rm log.txt


do_all: clear_all rm_log train generate


do_all_cfg: clear_all rm_log train_cfg generate_cfg


do_all_single:
	make rm_log clear_ckpt clear_gen train_cfg generate_cfg \
		EXP_NAME=${SINGLE_NAME} DB_NAME=${SINGLE_NAME} DATA_VERSION=${SINGLE_NAME} DB_VERSION=${SINGLE_NAME} USE_SAMPLE= \
		MTYPE=${MTYPE} TAB_TRAINER_CFG=${TAB_TRAINER_CFG} DEG_TRAINER_CFG=${DEG_TRAINER_CFG} \
		TAB_TRAIN_CFG=${TAB_TRAIN_CFG} DEG_TRAIN_CFG=${DEG_TRAIN_CFG}


cont_do_all_single:
	make rm_log train_cfg generate_cfg \
		EXP_NAME=${SINGLE_NAME} DB_NAME=${SINGLE_NAME} DATA_VERSION=${SINGLE_NAME} DB_VERSION=${SINGLE_NAME} USE_SAMPLE= \
		MTYPE=${MTYPE} TAB_TRAINER_CFG=${TAB_TRAINER_CFG} DEG_TRAINER_CFG=${DEG_TRAINER_CFG} \
		TAB_TRAIN_CFG=${TAB_TRAIN_CFG} DEG_TRAIN_CFG=${DEG_TRAIN_CFG}


alset_all: kill rm_log
	make prepare_all_alset
	make train_cfg generate_cfg EXP_NAME=unrelated \
		DATA_VERSION=full USE_SAMPLE= DB_VERSION=ur MTYPE=unrelated \
		TAB_TRAINER_CFG=ctgan DEG_TRAINER_CFG=mlp TAB_TRAIN_CFG=tab_custom DEG_TRAIN_CFG=deg_custom
	make kill
	make train_cfg generate_cfg EXP_NAME=pc \
		DATA_VERSION=full USE_SAMPLE= DB_VERSION=pc MTYPE=parent-child \
		TAB_TRAINER_CFG=ctgan DEG_TRAINER_CFG=mlp TAB_TRAIN_CFG=tab_custom DEG_TRAIN_CFG=deg_custom
	make kill
	make train_cfg generate_cfg EXP_NAME=ad \
		DATA_VERSION=full USE_SAMPLE= DB_VERSION=ad MTYPE=ancestor-descendant \
		TAB_TRAINER_CFG=ctgan DEG_TRAINER_CFG=mlp TAB_TRAIN_CFG=tab_custom DEG_TRAIN_CFG=deg_custom
	make kill
	make train_cfg generate_cfg EXP_NAME=default \
		DATA_VERSION=full USE_SAMPLE= DB_VERSION=af MTYPE=affecting \
		TAB_TRAINER_CFG=ctgan DEG_TRAINER_CFG=mlp TAB_TRAIN_CFG=tab_custom DEG_TRAIN_CFG=deg_custom
	make kill
	make train_cfg generate_cfg EXP_NAME=tvae_tab \
		DATA_VERSION=full USE_SAMPLE= DB_VERSION=af MTYPE=affecting \
		TAB_TRAINER_CFG=tvae DEG_TRAINER_CFG=mlp TAB_TRAIN_CFG=tab_custom DEG_TRAIN_CFG=deg_custom
	make kill
	make train_cfg generate_cfg EXP_NAME=mlp_tab \
		DATA_VERSION=full USE_SAMPLE= DB_VERSION=af MTYPE=affecting \
		TAB_TRAINER_CFG=mlp DEG_TRAINER_CFG=mlp TAB_TRAIN_CFG=tab_custom DEG_TRAIN_CFG=deg_custom
	make kill
	make train_cfg generate_cfg EXP_NAME=ctgan_deg \
		DATA_VERSION=full USE_SAMPLE= DB_VERSION=af MTYPE=affecting \
		TAB_TRAINER_CFG=ctgan DEG_TRAINER_CFG=ctgan TAB_TRAIN_CFG=tab_custom DEG_TRAIN_CFG=deg_custom
	make kill
	make train_cfg generate_cfg EXP_NAME=rvae_deg \
		DATA_VERSION=full USE_SAMPLE= DB_VERSION=af MTYPE=affecting \
		TAB_TRAINER_CFG=ctgan DEG_TRAINER_CFG=tvae TAB_TRAIN_CFG=tab_custom DEG_TRAIN_CFG=deg_custom
	make kill
	-mkdir -p ${BASE_DIR}/evaluate${OUT_SUFFIX}
	-mkdir -p ${BASE_DIR}/evaluate${OUT_SUFFIX}/alaset
	-python3.9 -W ignore main.py --log_level WARN --temp_cache .temp${OUT_SUFFIX} --num_processes 10 evaluate \
		--real_db_dir ${BASE_DIR}/model${OUT_SUFFIX}/alset/af/real_db \
		--fake_db_dir \
			${BASE_DIR}/generated${OUT_SUFFIX}/alset/unrelated/fake_db \
			${BASE_DIR}/generated${OUT_SUFFIX}/alset/pc/fake_db \
			${BASE_DIR}/generated${OUT_SUFFIX}/alset/ad/fake_db \
			${BASE_DIR}/generated${OUT_SUFFIX}/alset/default/fake_db \
			${BASE_DIR}/generated${OUT_SUFFIX}/alset/tvae_tab/fake_db \
			${BASE_DIR}/generated${OUT_SUFFIX}/alset/mlp_tab/fake_db \
			${BASE_DIR}/generated${OUT_SUFFIX}/alset/ctgan_deg/fake_db \
			${BASE_DIR}/generated${OUT_SUFFIX}/alset/tvae_deg/fake_db \
		--evaluator_path config/evaluator/alset_full.json \
		--evaluate_path config/evaluator/alset_full.json \
		--save_eval_res_to ${BASE_DIR}/evaluate${OUT_SUFFIX}/alset/full/results \
		--save_complete_result_to ${BASE_DIR}/evaluate${OUT_SUFFIX}/alset/full/complete \
		--save_tables_to ${BASE_DIR}/evaluate${OUT_SUFFIX}/alset/full/tables \
		--save_visualization_to ${BASE_DIR}/evaluate${OUT_SUFFIX}/alset/full/visualization \
		--save_all_res_to ${BASE_DIR}/evaluate${OUT_SUFFIX}/alset/full/result.pkl >> log.txt

