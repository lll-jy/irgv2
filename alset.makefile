prepare_small:
	python3.9 process.py database alset \
        --src_data_dir examples/data.nosync/alset \
        --data_dir examples/data.nosync/alset/samples \
        --meta_dir examples/alset/metadata/results \
        --out examples/data.nosync/alset/db_config.json \
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

train_small:
	-python3.9 -W ignore main.py --log_level WARN --num_processes 10 --temp_cache .temp.nosync train_gen \
        --db_config_path examples/data.nosync/alset/db_config.json \
        --data_dir examples/data.nosync/alset/samples \
        --db_dir_path examples/model.nosync/alset/small/real_db \
        --aug_resume \
        --default_tab_trainer_log_dir examples/model.nosync/alset/small/tf/tab \
        --default_deg_trainer_log_dir examples/model.nosync/alset/small/tf/deg \
        --default_tab_trainer_ckpt_dir examples/model.nosync/alset/small/ckpt/tab \
        --default_deg_trainer_ckpt_dir examples/model.nosync/alset/small/ckpt/deg \
        --skip_generate > log.txt
	du -sh .temp.nosync
	du -sh examples/model.nosync/alset/small

clear:
	-pkill -9 -f main.py
	-pkill -9 -f torch.multiprocessing.spawn
	-pkill -9 -f torch.multiprocessing.fork
	-rm -r .temp
	-rm -r .temp.nosync
	-rm -r examples/model.nosync/alset/small/
