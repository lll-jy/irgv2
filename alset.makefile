DB_NAME=alset
EXP_NAME=small
DATA_VERSION=samples
TAB_TRAINER_CFG=default
DEG_TRAINER_CFG=default
TAB_TRAIN_CFG=default
DEG_TRAIN_CFG=default
SCALING=1
EVALUATOR_CFG=default
EVALUATE_CFG=default
USE_SAMPLE=/samples
DB_VERSION=small
MTYPE=affecting

prepare_small_alset:
	python3.9 process.py database alset \
        --src_data_dir examples/data.nosync/alset \
        --data_dir examples/data.nosync/alset/samples \
        --meta_dir examples/alset/metadata/results \
        --out examples/data.nosync/alset/samples_db_config.json \
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


prepare_all_alset:
	echo TODO


prepare_table_alset:
	python3.9 process.py database alset \
        --src_data_dir examples/data.nosync/alset \
        --data_dir examples/data.nosync/alset/table \
        --meta_dir examples/alset/metadata/results \
        --out examples/data.nosync/alset/table_db_config.json \
        --redo_meta --redo_data \
        --tables \
            personal_data \
        --sample 50

train:
	-python3.9 -W ignore main.py --log_level WARN --num_processes 10 --temp_cache .temp.nosync train_gen \
        --db_config_path examples/data.nosync/${DB_NAME}/${DATA_VERSION}_db_config.json \
        --data_dir examples/data.nosync/${DB_NAME}/${DATA_VERSION}${USE_SAMPLE} \
        --db_dir_path examples/model.nosync/${DB_NAME}/${DB_VERSION}/real_db \
        --aug_resume \
        --default_tab_trainer_log_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/tf/tab \
        --default_deg_trainer_log_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/tf/deg \
        --default_tab_trainer_ckpt_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/ckpt/tab \
        --default_deg_trainer_ckpt_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/ckpt/deg \
        --skip_generate >> log.txt
	du -sh .temp.nosync
	du -sh examples/model.nosync/${DB_NAME}/${EXP_NAME}


train_cfg:
	-python3.9 -W ignore main.py --log_level WARN --num_processes 10 --temp_cache .temp.nosync train_gen \
        --db_config_path examples/data.nosync/${DB_NAME}/${DATA_VERSION}_db_config.json \
        --data_dir examples/data.nosync/${DB_NAME}/${DATA_VERSION}${USE_SAMPLE} \
        --db_dir_path examples/model.nosync/${DB_NAME}/${DB_VERSION}/real_db \
        --mtype ${MTYPE} \
        --aug_resume \
        --default_tab_trainer_args config/trainer/${TAB_TRAINER_CFG}.json \
        --default_deg_trainer_args config/trainer/${DEG_TRAINER_CFG}.json \
        --default_tab_train_args config/train/${TAB_TRAIN_CFG}.json \
        --default_deg_train_args config/train/${DEG_TRAIN_CFG}.json \
        --default_tab_trainer_log_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/tf/tab \
        --default_deg_trainer_log_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/tf/deg \
        --default_tab_trainer_ckpt_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/ckpt/tab \
        --default_deg_trainer_ckpt_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/ckpt/deg \
        --skip_generate >> log.txt
	du -sh .temp.nosync
	du -sh examples/model.nosync/${DB_NAME}/${EXP_NAME}


generate:
	-mkdir -p examples/generated.nosync/${DB_NAME}/${EXP_NAME}
	-python3.9 -W ignore main.py --log_level WARN --temp_cache .temp.nosync --num_processes 10 train_gen \
        --db_config_path examples/data.nosync/${DB_NAME}/${DATA_VERSION}_db_config.json \
        --data_dir examples/data.nosync/${DB_NAME}/${DATA_VERSION}${USE_SAMPLE} \
        --db_dir_path examples/model.nosync/${DB_NAME}/${DB_VERSION}/real_db \
        --aug_resume \
        --skip_train \
        --default_tab_train_resume True \
        --default_deg_train_resume True \
        --default_tab_trainer_log_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/tf/tab \
        --default_deg_trainer_log_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/tf/deg \
        --default_tab_trainer_ckpt_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/ckpt/tab \
        --default_deg_trainer_ckpt_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/ckpt/deg \
        --save_generated_to examples/generated.nosync/${DB_NAME}/${EXP_NAME}/generated \
        --save_synth_db examples/generated.nosync/${DB_NAME}/${EXP_NAME}/fake_db >> log.txt
	du -sh .temp.nosync
	du -sh examples/model.nosync/${DB_NAME}/${EXP_NAME}
	du -sh examples/generated.nosync/${DB_NAME}/${EXP_NAME}


generate_cfg:
	-mkdir -p examples/generated.nosync/${DB_NAME}/${EXP_NAME}
	-python3.9 -W ignore main.py --log_level WARN --temp_cache .temp.nosync --num_processes 10 train_gen \
        --db_config_path examples/data.nosync/${DB_NAME}/${DATA_VERSION}_db_config.json \
        --data_dir examples/data.nosync/${DB_NAME}/${DATA_VERSION}${USE_SAMPLE} \
        --db_dir_path examples/model.nosync/${DB_NAME}/${DB_VERSION}/real_db \
        --aug_resume \
		--default_tab_trainer_args config/trainer/${TAB_TRAINER_CFG}.json \
		--default_deg_trainer_args config/trainer/${DEG_TRAINER_CFG}.json \
		--default_tab_train_args config/train/${TAB_TRAIN_CFG}.json \
		--default_deg_train_args config/train/${DEG_TRAIN_CFG}.json \
        --skip_train \
        --default_tab_train_resume True \
        --default_deg_train_resume True \
        --default_scaling ${SCALING} \
        --default_tab_trainer_log_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/tf/tab \
        --default_deg_trainer_log_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/tf/deg \
        --default_tab_trainer_ckpt_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/ckpt/tab \
        --default_deg_trainer_ckpt_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/ckpt/deg \
        --save_generated_to examples/generated.nosync/${DB_NAME}/${EXP_NAME}/generated \
        --save_synth_db examples/generated.nosync/${DB_NAME}/${EXP_NAME}/fake_db >> log.txt
	du -sh .temp.nosync
	du -sh examples/model.nosync/${DB_NAME}/${EXP_NAME}
	du -sh examples/generated.nosync/${DB_NAME}/${EXP_NAME}


evaluate:
	-mkdir -p examples/evaluate.nosync
	-mkdir -p examples/evaluate.nosync/${DB_NAME}
	-mkdir -p examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}
	-python3.9  -W ignore main.py --log_level WARN --temp_cache .temp.nosync --num_processes 10 evaluate \
		--real_db_dir examples/model.nosync/${DB_NAME}/${DB_VERSION}/real_db \
		--fake_db_dir examples/generated.nosync/${DB_NAME}/${EXP_NAME}/fake_db \
		--save_eval_res_to examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}/results \
		--save_complete_result_to examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}/complete \
		--save_tables_to examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}/tables \
		--save_visualization_to examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}/visualization \
		--save_all_res_to examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}/result.pkl >> log.txt
	du -sh .temp.nosync
	du -sh examples/model.nosync/${DB_NAME}/${EXP_NAME}
	du -sh examples/generated.nosync/${DB_NAME}/${EXP_NAME}
	du -sh examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}


evaluate_cfg:
	-mkdir -p examples/evaluate.nosync
	-mkdir -p examples/evaluate.nosync/${DB_NAME}
	-mkdir -p examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}
	-python3.9  -W ignore main.py --log_level WARN --temp_cache .temp.nosync --num_processes 10 evaluate \
		--real_db_dir examples/model.nosync/${DB_NAME}/${DB_VERSION}/real_db \
		--fake_db_dir examples/generated.nosync/${DB_NAME}/${EXP_NAME}/fake_db \
		--evaluator_path config/evaluator/${EVALUATOR_CFG}.json \
		--evaluate_path config/evaluator/${EVALUATE_CFG}.json \
		--save_eval_res_to examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}/results \
		--save_complete_result_to examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}/complete \
		--save_tables_to examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}/tables \
		--save_visualization_to examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}/visualization \
		--save_all_res_to examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}/result.pkl >> log.txt
	du -sh .temp.nosync
	du -sh examples/model.nosync/${DB_NAME}/${EXP_NAME}
	du -sh examples/generated.nosync/${DB_NAME}/${EXP_NAME}
	du -sh examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}


kill:
	-pkill -9 -f main.py
	-pkill -9 -f torch.multiprocessing.spawn
	-pkill -9 -f torch.multiprocessing.fork


clear: kill
	-rm -r .temp
	-rm -r .temp.nosync
	-rm -r examples/model.nosync/${DB_NAME}/${EXP_NAME}/


clear_ckpt: kill
	-rm -r examples/model.nosync/${DB_NAME}/${EXP_NAME}/ckpt
	-rm -r examples/model.nosync/${DB_NAME}/${EXP_NAME}/tf


clear_gen: kill
	-rm -r examples/generated.nosync/${DB_NAME}/${EXP_NAME}/


clear_eval: kill
	-rm -r examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}/


clear_all: clear_eval clear_gen clear_ckpt clear


rm_log:
	-rm log.txt


do_all: clear_all rm_log train generate


do_all_cfg: clear_all rm_log train_cfg generate_cfg


alset_all:
	make -f alset.makefile prepare_all_alset
	make -f alset.makefile train_cfg generate_cfg EXP_NAME=unrelated \
		DATA_VERSION=full USE_SAMPLE= DB_VERSION=ur MTYPE=unrelated \
		TAB_TRAINER_CFG=ctgan DEG_TRAINER_CFG=mlp TAB_TRAIN_CFG=tab_custom DEG_TRAIN_CFG=deg_custom
	make -f alset.makefile train_cfg generate_cfg EXP_NAME=pc \
		DATA_VERSION=full USE_SAMPLE= DB_VERSION=pc MTYPE=parent-child \
		TAB_TRAINER_CFG=ctgan DEG_TRAINER_CFG=mlp TAB_TRAIN_CFG=tab_custom DEG_TRAIN_CFG=deg_custom
	make -f alset.makefile train_cfg generate_cfg EXP_NAME=ad \
		DATA_VERSION=full USE_SAMPLE= DB_VERSION=ad MTYPE=ancestor-descendant \
		TAB_TRAINER_CFG=ctgan DEG_TRAINER_CFG=mlp TAB_TRAIN_CFG=tab_custom DEG_TRAIN_CFG=deg_custom
	make -f alset.makefile train_cfg generate_cfg EXP_NAME=default \
		DATA_VERSION=full USE_SAMPLE= DB_VERSION=af MTYPE=affecting \
		TAB_TRAINER_CFG=ctgan DEG_TRAINER_CFG=mlp TAB_TRAIN_CFG=tab_custom DEG_TRAIN_CFG=deg_custom
	make -f alset.makefile train_cfg generate_cfg EXP_NAME=tvae_tab \
		DATA_VERSION=full USE_SAMPLE= DB_VERSION=af MTYPE=affecting \
		TAB_TRAINER_CFG=tvae DEG_TRAINER_CFG=mlp TAB_TRAIN_CFG=tab_custom DEG_TRAIN_CFG=deg_custom
	make -f alset.makefile train_cfg generate_cfg EXP_NAME=mlp_tab \
		DATA_VERSION=full USE_SAMPLE= DB_VERSION=af MTYPE=affecting \
		TAB_TRAINER_CFG=mlp DEG_TRAINER_CFG=mlp TAB_TRAIN_CFG=tab_custom DEG_TRAIN_CFG=deg_custom
	make -f alset.makefile train_cfg generate_cfg EXP_NAME=ctgan_deg \
		DATA_VERSION=full USE_SAMPLE= DB_VERSION=af MTYPE=affecting \
		TAB_TRAINER_CFG=ctgan DEG_TRAINER_CFG=ctgan TAB_TRAIN_CFG=tab_custom DEG_TRAIN_CFG=deg_custom
	make -f alset.makefile train_cfg generate_cfg EXP_NAME=rvae_deg \
		DATA_VERSION=full USE_SAMPLE= DB_VERSION=af MTYPE=affecting \
		TAB_TRAINER_CFG=ctgan DEG_TRAINER_CFG=tvae TAB_TRAIN_CFG=tab_custom DEG_TRAIN_CFG=deg_custom
	-mkdir -p examples/evaluate.nosync
	-mkdir -p examples/evaluate.nosync/alaset
	-python3.9 -W ignore main.py --log_level WARN --temp_cache .temp.nosync --num_processes 10 evaluate \
		--real_db_dir examples/model.nosync/alset/af/real_db \
		--fake_db_dir \
			examples/generated.nosync/alset/unrelated/fake_db \
			examples/generated.nosync/alset/pc/fake_db \
			examples/generated.nosync/alset/ad/fake_db \
			examples/generated.nosync/alset/default/fake_db \
			examples/generated.nosync/alset/tvae_tab/fake_db \
			examples/generated.nosync/alset/mlp_tab/fake_db \
			examples/generated.nosync/alset/ctgan_deg/fake_db \
			examples/generated.nosync/alset/tvae_deg/fake_db \
		--evaluator_path config/evaluator/alset_full.json \
		--evaluate_path config/evaluator/alset_full.json \
		--save_eval_res_to examples/evaluate.nosync/alset/full/results \
		--save_complete_result_to examples/evaluate.nosync/alset/full/complete \
		--save_tables_to examples/evaluate.nosync/alset/full/tables \
		--save_visualization_to examples/evaluate.nosync/alset/full/visualization \
		--save_all_res_to examples/evaluate.nosync/alset/full/result.pkl >> log.txt

