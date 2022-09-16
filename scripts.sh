# ALSET
## Prepare
### Small
python3.9 process.py database alset \
    --src_data_dir examples/data.nosync/alset \
    --data_dir examples/data.nosync/alset/processed \
    --meta_dir examples/alset/metadata/results \
    --out examples/data.nosync/alset/db_config.json \
    --redo_meta --redo_data \
    --tables \
        personal_data \
        sis_academic_program_offer \
        sis_academic_program \
        sis_academic_career sis_plan_offer \
        sis_academic_plan \
        sis_enrolment \
    --sample 50

## Train
python3.9 -W ignore main.py --log_level WARN train_gen \
  --db_config_path examples/data.nosync/alset/db_config.json \
  --data_dir examples/data.nosync/alset/samples \
  --db_dir_path examples/model.nosync/alset/small/real_db \
  --aug_resume \
  --default_tab_trainer_log_dir examples/model.nosync/alset/small/tf/tab \
  --default_deg_trainer_log_dir examples/model.nosync/alset/small/tf/deg \
  --default_tab_trainer_ckpt_dir examples/model.nosync/alset/small/ckpt/tab \
  --default_deg_trainer_ckpt_dir examples/model.nosync/alset/small/ckpt/deg \
  --skip_generate
