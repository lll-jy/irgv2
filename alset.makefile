DB_NAME=alset
EXP_NAME=small
DATA_VERSION=samples

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

train:
	-python3.9 -W ignore main.py --log_level WARN --num_processes 10 --temp_cache .temp.nosync train_gen \
        --db_config_path examples/data.nosync/${DB_NAME}/${DATA_VERSION}_db_config.json \
        --data_dir examples/data.nosync/${DB_NAME}/${DATA_VERSION} \
        --db_dir_path examples/model.nosync/${DB_NAME}/${EXP_NAME}/real_db \
        --aug_resume \
        --default_tab_trainer_log_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/tf/tab \
        --default_deg_trainer_log_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/tf/deg \
        --default_tab_trainer_ckpt_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/ckpt/tab \
        --default_deg_trainer_ckpt_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/ckpt/deg \
        --skip_generate > log.txt
	du -sh .temp.nosync
	du -sh examples/model.nosync/${DB_NAME}/${EXP_NAME}


generate:
	-mkdir -p examples/generated.nosync/${DB_NAME}/${EXP_NAME}
	-python3.9 -W ignore main.py --log_level WARN --temp_cache .temp.nosync --num_processes 10 train_gen \
        --db_config_path examples/data.nosync/${DB_NAME}/${DATA_VERSION}_db_config.json \
        --data_dir examples/data.nosync/${DB_NAME}/${DATA_VERSION} \
        --db_dir_path examples/model.nosync/${DB_NAME}/${EXP_NAME}/real_db \
        --aug_resume \
        --skip_train \
        --default_tab_train_resume True \
        --default_deg_train_resume True \
        --default_tab_trainer_log_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/tf/tab \
        --default_deg_trainer_log_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/tf/deg \
        --default_tab_trainer_ckpt_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/ckpt/tab \
        --default_deg_trainer_ckpt_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/ckpt/deg \
        --save_generated_to examples/generated.nosync/${DB_NAME}/${EXP_NAME}/generated \
        --save_synth_db examples/generated.nosync/${DB_NAME}/${EXP_NAME}/fake_db > log.txt
	du -sh .temp.nosync
	du -sh examples/model.nosync/${DB_NAME}/${EXP_NAME}
	du -sh examples/generated.nosync/${DB_NAME}/${EXP_NAME}


evaluate:
	-mkdir -p examples/evaluate.nosync
	-mkdir -p examples/evaluate.nosync/${DB_NAME}
	-mkdir -p examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}
	-python3.9  -W ignore main.py --log_level WARN --temp_cache .temp.nosync --num_processes 10 evaluate \
		--real_db_dir examples/model.nosync/${DB_NAME}/${EXP_NAME}/real_db \
		--fake_db_dir examples/generated.nosync/${DB_NAME}/${EXP_NAME}/fake_db \
		--save_eval_res_to examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}/trivial \
		--save_complete_result_to examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}/complete \
		--save_synthetic_tables_to examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}/tables/synthetic \
		--save_tables_to examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}/tables/real \
		--save_visualization_to examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}/visualization \
		--save_all_res_to examples/evaluate.nosync/${DB_NAME}/${EXP_NAME}/result #> log.txt
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
