# Incremental Relational Generator

## Group 1
* Arguments
  * `EXP_NAME`=degree_full
  * `DB_VERSION`=model_compare
  * `DATA_VERSION`=model_compare
  * configuration
    * tabular: ctgan
      * `embedding_dim`=50
      * `g/d_dim`=256, 256
      * `lr`=2e-4,
      * `wd`: 0
    * degree: stepped
    * train:
      * `epochs`=10
      * `batch_size`=500
* Tables
  * `personal_data`: 5000 students, 5000x6
  * `sis_milestone`: these 500 students, 14917x15
  * `module_offer`: not synthetic, 2500 modules from the 5000 students, 2500x5
  * `module_enrolment`: from the students and modules, 53543x6
  * `uci_gym`: from the 5000 students, 8358x6
* Generated
  * : simple
  * `_half`: all 0.5
  * `_half_but_mod`: all 0.5, except module_offer 1
  * `_double`: all 2
  * `_ununiform`: module_enrolment 0.5, sis_milestone 1.5

## Group 2
* Arguments
  * `EXP_NAME`=wifi
  * `DB_VERSION`=gen_wifi
  * `DATA_VERSION`=gen_wifi
* gen_wifi