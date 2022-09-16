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
