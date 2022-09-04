from datetime import datetime
from functools import partial
from typing import Dict, Set, Any, Type, Union

import numpy as np
import pandas as pd


"""
Ignored:
'company': {
    2010: 'ans_11',
}
'occupation': {
    2010: 'ans_13',
    2011: 'q11',
    2012: 'q16',
    2013: 'q12',
    2014: 'q12',
    2015: 'q12',
    2016: 'occupation',
    2017: 'q12',
    2018: 'q12',
    2019: 'q12',
    2020: 'q12'
}
"""

_GES_COLUMNS: Dict[str, Dict[int, str]] = {
    'student_token': {
        2010: 'student_token',
        2011: 'student_token',
        2012: 'student_token',
        2013: 'student_token',
        2014: 'student_token',
        2015: 'student_token',
        2016: 'student_token',
        2017: 'student_token',
        2018: 'student_token',
        2019: 'student_token',
        2020: 'student_token'
    },
    'academic_load': {
        2010: 'type',
        2011: 'academic_load',
        2012: 'type',
        2013: 'graduate_type',
        2014: 'graduate_type',
        2015: 'graduate_type',
        2016: 'segment',
        2017: 'academicloaddescription',
        2018: 'segment',
        2019: 'segment',
        2020: 'acadeload'
    },
    'gender': {
        2010: 'gender',
        2011: 'gender_coded_org',
        2012: 'gender',
        2013: 'gender',
        2014: 'gender',
        2015: 'gender',
        2016: 'gender',
        2017: 'gender',
        2018: 'gender',
        2019: 'gender',
        2020: 'gender'
    },
    'degree': {
        2010: 'degree',
        2011: 'degree',
        2012: 'degree',
        2013: 'degree',
        2014: 'nus_programme',
        2015: 'degree',
        2016: 'conferred_degree_1st_major',
        2017: 'conferreddegreedescr',
        2018: 'conferreddegreedescr',
        2019: 'degree',
        2020: 'degreeconferred'
    },
    'major': {
        2010: 'major1',
        2011: 'major1_org',
        2012: 'major1',
        2013: 'major_1',
        2014: 'major1_1',
        2015: 'major',
        2016: 'honours_plan_1st_major',
        2017: 'major1',
        2018: 'major1',
        2019: 'major1ascapturedingraduatingsemester',
        2020: 'major1descr'
    },
    'usp': {
        2010: 'usp',
        2011: 'usp',
        2012: 'usp',
        2013: 'usp',
        2014: 'nus_usp',
        2015: 'nus_usp',
        2016: 'usp',
        2017: 'usp',
        2018: 'usp',
        2019: 'usp'
    },
    'faculty': {
        2010: 'faculty',
        2011: 'faculty_coded_org',
        2012: 'faculty',
        2013: 'college',
        2014: 'nus_faculty',
        2015: 'faculty',
        2016: 'faculty_1st_major',
        2017: 'facultydescr',
        2018: 'facultydescr',
        2019: 'faculty',
        2020: 'facultydescr'
    },
    'noc': {
        2010: 'noc',
        2011: 'onnocduringcandidature_org',
        2012: 'noc',
        2013: 'noc',
        2014: 'nus_noc',
        2015: 'nus_noc',
        2016: 'noc',
        2017: 'noc',
        2018: 'noc',
        2019: 'nocgraduateyorn'
    },
    'segment_als': {
        2010: 'segment_als',
        2011: 'segment_als',
        2012: 'segment_als',
        2013: 'segment_als',
        2014: 'segment_als',
        2015: 'segment_als',
        2016: 'segment_als',
        2017: 'segment_als',
        2018: 'segment_als',
        2019: 'segment_als',
        2020: 'segment_als'
    },
    'hon': {
        2010: 'hon',
        2011: 'degreetype_org',
        2012: 'hon',
        2013: 'awardshortform_recode1',
        2014: 'nus_hon',
        2015: 'honours',
        2016: 'hon',
        2017: 'dv13',
        2018: 'hon',
        2019: 'hon',
        2020: 'hon'
    },
    'class_of_hons': {
        2011: 'class_of_hons',
        2012: 'class',
        2013: 'award_string1',
        2014: 'nus_award1',
        2015: 'nus_honoursplan',
        2016: 'honours_class_1st_major',
        2017: 'honours_descr',
        2018: 'honours_descr',
        2019: 'class_of_hons'
    },
    'confer_date': {
        2011: 'conferreddate_org',
        2012: 'confer_date',
        2013: 'conferreddate_recode',
        2014: 'conferreddate',
        2015: 'conferreddate',
        2016: 'conferred_date',
        2017: 'conferreddate',
        2018: 'conferreddate',
        2019: 'conferreddate',
        2020: 'confermentdate'
    },
    'progtype': {
        2012: 'progtype',
        2013: 'progtype',
        2014: 'nus_progtype',
        2015: 'nus_progtype',
        2016: 'programme_type',
        2017: 'progtype',
        2018: 'progtype',
        2019: 'progtype',
        2020: 'programtype'
    },
    'first_choice': {
        2012: 'choice',
        2013: 'choice_yes',
        2015: 'choice_course',
        2016: '1st_choice_of_study',
        2017: '1stchoicecourseofstudy',
        2018: '1stchoicecourseofstudy',
        2019: '1stchoicecourseofstudy'
    },
    'is_first_choice': {
        2012: 'choice_yes',
        2013: 'choice',
        2015: 'choice',
        2016: '1st_choice_of_study_as_current_course_recoded',
        2019: 'choiceofstudy'
    },
    'professional_degree': {
        2012: 'professionaldegree',
        2013: 'professional_degree',
        2014: 'programme_type'
    },
    'residency': {
        2010: 'ans_01',
        2011: 'q1',
        2012: 'd2',
        2013: 'residency',
        2014: 'd2',
        2015: 'd2',
        2016: 'residency',
        2017: 'residencydescr',
        2018: 'residencydescr',
        2019: 'residency',
        2020: 'd1'
    },
    'fin_src': {
        2010: 'ans_05',
        2011: 'q3',
        2012: 'q40',
        2013: 'q40',
        2014: 'nus_q41'
    },
    'race': {
        2015: 'race'
    },
    'new_citizen': {
        2010: 'ans_02',
        2011: 'q2',
        2012: 'd3',
        2015: 'd3'
    },
    'primary_school_in_sg': {
        2010: 'ans_03_1'
    },
    'secondary_school_in_sg': {
        2010: 'ans_03_2'
    },
    'ite_in_sg': {
        2010: 'ans_03_3'
    },
    'jc_mi_in_sg': {
        2010: 'ans_03_4'
    },
    'poly_in_sg': {
        2010: 'ans_03_5'
    },
    'none_edu_in_sg': {
        2010: 'ans_03_6'
    },
    'final_exam': {
        2010: 'final_exam'
    },
    'activity_status': {
        2010: 'activity_status',
        2011: 'activity_status',
        2012: 'q1',
        2013: 'q1',
        2014: 'q1',
        2015: 'q1',
        2016: 'activity_status',
        2017: 'q1',
        2018: 'q1',
        2019: 'q1',
        2020: 'q1_b'
    },
    'not_ft_perm_reason': {
        2010: 'ans_08',
        2011: 'q6',
        2012: 'q3',
        2013: 'q3',
        2014: 'q3',
        2015: 'q3',
        2016: 'main_reason_for_working_part_time_temporary_freelance',
        2017: 'q3',
        2018: 'q3',
        2019: 'q3',
        2020: 'q3'
    },
    'employment_status': {
        2010: 'ans_09a',
        2011: 'q7',
        2012: 'q4',
        2013: 'q4',
        2014: 'q4',
        2015: 'q4',
        2016: 'employment_status',
        2017: 'q4',
        2018: 'q4',
        2019: 'q4',
        2020: 'q4'
    },
    'company_type': {
        2010: 'ans_10',
        2011: 'q8',
        2012: 'q5',
        2013: 'q5',
        2014: 'q5',
        2015: 'q5',
        2016: 'ogranisation_type_recoded',
        2017: 'q5',
        2018: 'q5',
        2019: 'q5',
        2020: 'q5'
    },
    'industry': {
        2010: 'ans_12',
        2011: 'q10',
        2012: 'q15',
        2013: 'q11',
        2014: 'q11',
        2025: 'q11',
        2016: 'industry',
        2017: 'q11',
        2018: 'q11',
        2019: 'q11',
        2020: 'q11'
    },
    'basic_salary': {
        2010: 'ans_14',
        2011: 'q12a',
        2012: 'q14_basic',
        2013: 'q10_basic',
        2014: 'q10_basic',
        2015: 'q10_basic',
        2016: 'basic_monthly_salary',
        2017: 'q10_basic',
        2018: 'q10_basic',
        2019: 'q10_basic',
        2020: 'q10_basic'
    },
    'gross_salary': {
        2011: 'q12b',
        2012: 'q14_sum',
        2013: 'q10_sum',
        2014: 'q10_gross',
        2015: 'q10_gross',
        2016: 'gross_salary'
    },
    'ot_salary': {
        2012: 'q14_ot',
        2013: 'q10_ot',
        2014: 'q10_ot',
        2015: 'q10_ot',
        2016: 'additional_pay_components',
        2017: 'q10_overtime',
        2018: 'q10_overtime',
        2019: 'q10_overtime',
        2020: 'q10_overtime'
    },
    'start_search': {
        2012: 'start_search',
        2013: 'q20_date',
        2014: 'start_search',
        2015: 'q20',
        2016: 'start_search',
        2017: 'q20_date',
        2018: 'q22_date',
        2019: 'start_search',
        2020: 'start_search'
    },
    'gap_grad_search': {
        2012: 'r_mth_q8_q7',
        2014: 'q20',
        2015: 'q20_recode',
        2016: 'gap_grad_search'
    },
    'offer_date': {
        2012: 'offer_date',
        2013: 'q19_date',
        2014: 'offer_date',
        2015: 'q19',
        2016: 'offer_date',
        2017: 'q19_date',
        2018: 'q25_date',
        2019: 'offer_date',
        2020: 'offer_date'
    },
    'offer_count': {
        2010: 'offer_count',
        2011: 'offer_count',
        2012: 'offer_count',
        2013: 'q18',
        2014: 'q18',
        2015: 'q18',
        2016: 'number_of_ftp_offers',
        2017: 'q18',
        2018: 'q24',
        2019: 'q24',
        2020: 'q18_b'
    },
    'offer_wait': {
        2010: 'ans_20',
        2011: 'q15',
        2012: 'offer_wait',
        2013: 'offer_wait',
        2014: 'q19_recode',
        2017: 'dv10_code',
        2018: 'dv10_code',
        2019: 'dv10_code'
    },
    'main_channel': {
        2010: 'ans_21',
        2011: 'q16',
        2012: 'q18',
        2013: 'q21',
        2014: 'nus_q21',
        2015: 'nus_q21',
        2016: 'ranked_1st'
    },
    'channel_nus_portal': {
        2016: 'cfg_job_portal',
        2017: 'q21_1',
        2018: 'q23_1',
        2019: 'q23_1',
        2020: 'q44_1'
    },
    'channel_cfg_posting': {
        2016: 'channel_services',
        2017: 'q21_2',
        2018: 'q23_2',
        2019: 'q23_2',
        2020: 'q44_2'
    },
    'channel_nus_events': {
        2016: 'cfg_career_events',
        2017: 'q21_3',
        2018: 'q23_3',
        2019: 'q23_3',
        2020: 'q44_3'
    },
    'channel_ia': {
        2016: 'channel_ia',
        2017: 'q21_4',
        2018: 'q23_4',
        2019: 'q23_4',
        2020: 'q44_4'
    },
    'channel_nus_ref': {
        2017: 'q21_5',
        2018: 'q23_5',
        2019: 'q23_5',
        2020: 'q44_5'
    },
    'channel_staff_ref': {
        2016: 'faculty_or_staff_referral',
        2017: 'q21_6',
        2018: 'q23_6',
        2019: 'q23_6',
        2020: 'q44_6'
    },
    'channel_fr_ref': {
        2016: 'referrals_by_family_or_friends',
        2017: 'q21_7',
        2018: 'q23_7',
        2019: 'q23_7',
        2020: 'q44_7'
    },
    'channel_other_web': {
        2016: 'other_job_portals',
        2017: 'q21_8',
        2018: 'q23_8',
        2019: 'q23_8',
        2020: 'q44_8'
    },
    'channel_social_media': {
        2016: 'social_media',
        2017: 'q21_9',
        2018: 'q23_9',
        2019: 'q23_9',
        2020: 'q44_9'
    },
    'channel_agency': {
        2016: 'recruitment_agencies',
        2017: 'q21_10',
        2018: 'q23_10',
        2019: 'q23_10',
        2020: 'q44_10'
    },
    'channel_traditional': {
        2016: 'printed_advertisements',
        2017: 'q21_11',
        2018: 'q23_11',
        2019: 'q23_11',
        2020: 'q44_11'
    },
    'channel_ext_cf': {
        2016: 'other_career_fairs',
        2017: 'q21_12',
        2018: 'q23_12',
        2019: 'q23_12',
        2020: 'q44_12'
    },
    'channel_no_need': {
        2018: 'q23_13',
        2019: 'q23_13',
        2020: 'q44_13'
    },
    'channel_other': {
        2016: 'other_job_channels',
        2017: 'q21_other',
        2018: 'q23_14',
        2019: 'q23_14',
    },
    'working_country': {
        2010: 'ans_22',
        2011: 'q20',
        2012: 'q10',
        2014: 'q7',
        2017: 'q7',
        2018: 'q7',
        2019: 'q7',
        2020: 'q7'
    },
    'is_overseas': {
        2012: 'q10_recode',
        2013: 'q7_recode',
        2015: 'q7_recode',
        2016: 'overseas_organisation'
    },
    'overseas_type': {
        2010: 'ans_23',
        2011: 'q21',
        2012: 'q11',
        2016: 'overseas_organisation',
        2017: 'q8',
        2018: 'q8',
        2019: 'q8',
        2020: 'q8'
    },
    'return_sg': {
        2010: 'ans_24',
        2011: 'q22',
        2012: 'q12',
        2016: 'intention_to_return_for_work',
        2017: 'q9',
        2018: 'q9',
        2019: 'q9',
        2020: 'q9',
    },
    'helped_by_course': {
        2010: 'ans_25a',
        2011: 'q17a',
        2012: 'q19a',
        2013: 'q13a',
        2014: 'q13a',
        2015: 'q13a',
        2016: 'helpfulness_of_course',
        2017: 'q13a',
        2018: 'q15a',
        2019: 'q15_a',
        2020: 'q13_a'
    },
    'helped_by_nus_brand': {
        2010: 'ans_25b',
        2011: 'q17b',
        2012: 'q19b',
        2013: 'q13b',
        2014: 'q13b',
        2015: 'q13a',
        2016: 'helpfulness_of_nus',
        2017: 'q13b',
        2018: 'q15b',
        2019: 'q15_b',
        2020: 'q13_b'
    },
    'course_related': {
        2010: 'ans_26',
        2011: 'q18',
        2012: 'q20',
        2013: 'q14',
        2014: 'q14',
        2015: 'q14',
        2016: 'relevance_to_course_of_study',
        2017: 'q14',
        2018: 'q16',
        2019: 'q16',
        2020: 'q14'
    },
    'unrelated_reason': {
        2010: 'ans_27',
        2011: 'q19',
        2012: 'q21',
        2013: 'q15',
        2014: 'q15',
        2015: 'q15',
        2016: 'main_reason_for_lack_of_relevance',
        2017: 'q15',
        2020: 'q15'
    },
    'unrelated_lack': {
        2018: 'q17_1',
        2019: 'q17_1'
    },
    'unrelated_failed': {
        2018: 'q17_2',
        2019: 'q17_2'
    },
    'unrelated_interest': {
        2018: 'q17_3',
        2019: 'q17_3'
    },
    'unrelated_strength': {
        2018: 'q17_4',
        2019: 'q17_4'
    },
    'unrelated_pay': {
        2018: 'q17_5',
        2019: 'q17_5'
    },
    'unrelated_opportunity': {
        2018: 'q17_6',
        2019: 'q17_6'
    },
    'unrelated_prospects': {
        2018: 'q17_7',
        2019: 'q17_7'
    },
    'unrelated_co_worker': {
        2018: 'q17_8',
        2019: 'q17_8'
    },
    'unrelated_balance': {
        2018: 'q17_9',
        2019: 'q17_9'
    },
    'unrelated_convenience': {
        2018: 'q17_10',
        2019: 'q17_10'
    },
    'unrelated_social_status': {
        2018: 'q17_11',
        2019: 'q17_11'
    },
    'unrelated_support': {
        2018: 'q17_12',
        2019: 'q17_12'
    },
    'unrelated_environment': {
        2018: 'q17_13',
        2019: 'q17_13'
    },
    'unrelated_others': {
        2018: 'q17_14',
        2019: 'q17_14'
    },
    'attend_career': {
        2010: 'ans_30_1',
        2011: 'q40a',
        2012: 'q32_1',
        2013: 'q27a_1'
    },
    'attend_faculty': {
        2010: 'ans_30_2',
        2011: 'q40b',
        2012: 'q32_2',
        2013: 'q27a_2'
    },
    'prepare_written_comm': {
        2010: 'ans_32a',
        2011: 'q41a',
        2012: 'q33a',
        2013: 'q33a',
        2014: 'q34a',
        2015: 'q34a',
        2016: 'preparation_written_communication',
        2017: 'q31_a',
        2018: 'q37_a',
        2019: 'q39_a',
        2020: 'q51_a'
    },
    'prepare_oral_comm': {
        2010: 'ans_32b',
        2011: 'q41b',
        2012: 'q33b',
        2013: 'q33b',
        2014: 'q34b',
        2015: 'q34b',
        2016: 'preparation_oral_communication_and_presentation',
        2017: 'q31_b',
        2018: 'q37_b',
        2019: 'q39_b',
        2020: 'q51_b'
    },
    'prepare_multidisciplinary': {
        2010: 'ans_32c',
        2011: 'q41c',
        2012: 'q33c',
        2013: 'q33c',
        2014: 'q34c',
        2015: 'q34c',
        2016: 'preparation_multidisciplinary',
        2017: 'q31_c',
        2018: 'q37_c',
        2019: 'q39_c',
        2020: 'q51_c'
    },
    'prepare_international': {
        2010: 'ans_32d',
        2011: 'q41d',
        2012: 'q33d',
        2013: 'q33d',
        2014: 'q34d',
        2015: 'q34d',
        2016: 'preparation_international',
        2017: 'q31_d',
        2018: 'q37_d',
        2019: 'q39_d',
        2020: 'q51_d'
    },
    'prepare_org': {
        2010: 'ans_32e',
        2011: 'q41e',
        2012: 'q33e',
        2013: 'q33e',
        2014: 'q34e',
        2015: 'q34e',
        2016: 'preparation_planning_and_organising',
        2017: 'q31_e',
        2018: 'q37_e',
        2019: 'q39_e',
        2020: 'q51_e'
    },
    'prepare_critical': {
        2010: 'ans_32f',
        2011: 'q41f',
        2012: 'q33f',
        2013: 'q33f',
        2014: 'q34f',
        2015: 'q34f',
        2016: 'preparation_critical_thinking_and_problem_solving',
        2017: 'q31_f',
        2018: 'q37_f',
        2019: 'q39_f',
        2020: 'q51_f'
    },
    'prepare_creative': {
        2010: 'ans_32g',
        2011: 'q41g',
        2012: 'q33g',
        2013: 'q33g',
        2014: 'q34g',
        2015: 'q34g',
        2016: 'preparation_creativity_and_innovation',
        2017: 'q31_g',
        2018: 'q37_g',
        2019: 'q39_g',
        2020: 'q51_g'
    },
    'prepare_learn_ind': {
        2010: 'ans_32h',
        2011: 'q41h',
        2012: 'q33h',
        2013: 'q33h',
        2014: 'q34h',
        2015: 'q34h',
        2016: 'preparation_independent_learning',
        2017: 'q31_h',
        2018: 'q37_h',
        2019: 'q39_h',
        2020: 'q51_h'
    },
    'prepare_interpersonal': {
        2010: 'ans_32i',
        2011: 'q41i',
        2012: 'q33i',
        2013: 'q33i',
        2014: 'q34i',
        2015: 'q34i',
        2016: 'preparation_interpersonal_effectiveness',
        2017: 'q31_i',
        2018: 'q37_i',
        2019: 'q39_i',
        2020: 'q51_i'
    },
    'prepare_personal': {
        2010: 'ans_32j',
        2011: 'q41j',
        2012: 'q33j',
        2013: 'q33j',
        2014: 'q34j',
        2015: 'q34j',
        2016: 'preparation_personal_effectiveness',
        2017: 'q31_j',
        2018: 'q37_j',
        2019: 'q39_j',
        2020: 'q51_j'
    },
    'prepare_cross_cultural': {
        2013: 'q33k'
    },
    'prepare_change_env': {
        2013: 'q33m'
    },
    'prepare_career': {
        2014: 'q34k',
        2015: 'q34k'
    },
    'prepare_domain': {
        2016: 'preparation_domain_expertise',
        2017: 'q31_k',
        2018: 'q39_k',
        2019: 'q51_k'
    }
}
_GENERAL_BINARY_EQUIV: Dict[str, Set[str]] = {
    'Y': {'1', '1.0', 'Y', 'Yes'},
    'N': {'2', '2.0', 'N', 'No'},
    'Nil': {'.', 'Not', 'Not applicable based on skipping pattern', 'Non Applicable based on skip pattern'}
}
_GENERAL_INC_RATE_EQUIV: Dict[str, Set[str]] = {
    '1': {'1', '1.0', 'Not at all'},
    '2': {'2', '2.0', 'A Little'},
    '3': {'3', '3.0', 'Fairly Well', 'To some extent'},
    '4': {'4', '4.0', 'Well'},
    '5': {'5', '5.0', 'Very Well', 'To a great extent'},
    'Nil': {'No comment', '.', 'Non Applicable based on skip pattern', 'Not applicable based on skipping pattern',
            'Not applicable ba'}
}
_GES_COLUMN_VALUE_EQUIV: Dict[str, Dict[Any, Set]] = {
    'academic_load': {
        'Full-Time': {'Full-Time', 'Full Time', '1', '1.0', 'Full-Time2', 'Full-Time3'},
        'Part-Time': {'Part-Time', 'Part Time', '2', '2.0'},
        'Follow-Up': {'Follow-Up', 'Follow Up', 'Follow up'}
    },
    'gender': {
        'M': {'M', 'Male', '1', '1.0'},
        'F': {'F', 'Female', '2', '2.0'}
    },
    'degree': {
        'Bachelor of Arts': {
            'B.A.', 'B.A. (Hons.)', 'Arts', 'Arts [Hons]', 'Bachelor of Arts', 'Bachelor of Arts (Hons)',
            'Bachelor of Arts (Hon)', 'Bachelor of Arts with Honours', '1', '2', '1.0', '2.0'
        },
        'Bachelor of Social Sciences': {
            'B.Soc.Sci. (Hons.)', 'Social Sciences [Hons]', 'Bachelor of Social Sciences (Hons)',
            'Bachelor of Social Sciences', '3', '4', '3.0', '4.0'
        },
        'Bachelor of Science': {
            'B.Sc. (Hons.)', 'B.Sc.', 'Science [Hons]', 'Science', 'Bachelor of Science (Hons)', 'Bachelor of Science',
            'Bachelor of Science (Hon)', 'Bachelor of Science with Honours', '17', '18', '20', '17.0', '18.0', '19.0'
        },
        'Bachelor of Engineering (Electrical Engineering)': {
            'B.Eng. (Elect.)', 'Electrical Engineering', 'Bachelor of Engineering (Electrical Engineering)',
            'Bachelor of Engineering (Electric', '9', '9.0'
        },
        'Bachelor of Business Administration': {
            'B.B.A.', 'B.B.A. (Hons.)', 'Business Administration [Honours]', 'Business Administration [3-yr programme]',
            'Bachelor of Business Administration (Hons)', 'Bachelor of Business Administration',
            'Bachelor of Business Administrati', 'Bachelor of Business Administration (Hon)', '22', '23', '22.0', '23.0'
        },
        'Bachelor of Engineering (Mechanical Engineering)': {
            'B.Eng. (Mech.)', 'Mechanical Engineering', 'Bachelor of Engineering (Mechanical Engineering)',
            'Bachelor of Engineering (Mechanic', '14', '14.0'
        },
        'Bachelor of Engineering (Chemical Engineering)': {
            'B.Eng. (Chem.)', 'Chemical Engineering', 'Bachelor of Engineering (Chemical Engineering)',
            'Bachelor of Engineering (Chemical', '6', '6.0'
        },
        'Bachelor of Laws': {
            'LL.B. (Hons.)', 'Law', 'Follow-Up: Law', 'Bachelor of Laws ', 'Bachelor of Laws (LLB) (Hons)', '15', '47',
            '15.0', '47.0'
        },
        'Bachelor of Science (Real Estate)': {
            'B.Sc. (Real Est.)', 'Real Estate', 'Bachelor of Science (Real Estate)', '36', '36.0'
        },
        'Bachelor of Medicine and Bachelor of Surgery': {
            'M.B.B.S.', 'Medicine and Bachelor of Surgery', 'Follow-Up: Medicine and Bachelor of Surgery',
            'Bachelor of Medicine and Bachelor of Surgery', 'Bachelor of Medicine and Bachelor', '39', '48', '39.0',
            '48.0'
        },
        'Bachelor of Science (Project and Facilities Management)': {
            'B.Sc. (Proj. & Facilities Mgt.)', 'Project and Facilities Management',
            'Bachelor of Science (Project and Facilities Management)', 'Bachelor of Science (Project and',
            'Bachelor of Science (Project and Facilities Manag', 'B.Sc. (Bldg.)', '35', '35.0'
        },
        'Bachelor of Science (Pharmacy)': {
            'B.Sc. (Pharm.) (Hons.)', 'Pharmacy', 'Follow-Up: Pharmacy', 'Bachelor of Science (Pharmacy) (Hons)',
            'Bachelor of Science (Pharmacy)', '21', '49', '21.0', '49.0'
        },
        'Bachelor of Technology (Electronics Engineering)': {
            'B. Tech. (Electronics Eng.)', 'Part-Time: B Tech [Electronics Eng]',
            'Bachelor of Technology (Electronics Engineering)', '42', '42.0'
        },
        'Bachelor of Computing (Information Systems)': {
            'B.Comp. (I.S.)', 'Information Systems', 'Bachelor of Computing (Information Systems)',
            'Bachelor of Computing (Informatio', '32', '32.0'
        },
        'Bachelor of Engineering (Computer Engineering)': {
            'B.Eng. (Comp.)', 'B.Comp. (Comp.Eng.)', 'Computer Engineering',
            'Bachelor of Engineering (Computer Engineering)', 'Bachelor of Engineering (Computer',
            'Bachelor of Computing (Computer Engineering)', '8', '29', '8.0', '29.0'
        },
        'Bachelor of Business Administration (Accountancy)': {
            'B.B.A. (Accountancy)', 'B.B.A. (Accountancy) (Hons.)', 'Business Administration [Accountancy]',
            'Business Administration [Accountancy] [Honours]', 'Bachelor of Business Administration (Accountancy)',
            'Bachelor of Business Administration (Accountancy) (Hons)',
            'Bachelor of Business Administration (Accountancy) (Hon)', '24', '25', '24.0', '25.0'
        },
        'Bachelor of Engineering (Civil Engineering)': {
            'B.Eng. (Civil)', 'Civil Engineering', 'Bachelor of Engineering (Civil Engineering)', '7', '7.0'
        },
        'Bachelor of Engineering (Industrial and Systems Engineering)': {
            'B.Eng. (ISE.)', 'Industrial and Systems Engineering',
            'Bachelor of Engineering (Industrial and Systems Engineering)', 'Bachelor of Engineering (Industri',
            'Bachelor of Engineering (Industrial and Systems E', '12', '12.0'
        },
        'Bachelor of Engineering (Biomedical Engineering)': {
            'B.Eng. (Bioengineering)', 'Bioengineering', 'Bachelor of Engineering (Bioengineering)',
            'Bachelor of Engineering (Biomedical Engineering)', 'Bachelor of Engineering (Biomedic', '5', '5.0'
        },
        'Bachelor of Computing (Electronic Commerce)': {
            'B.Comp. (E.Commerce)', 'Electronic Commerce', 'Bachelor of Computing (Electronic Commerce)',
            'Bachelor of Computing (Electronic', '31', '31.0'
        },
        'Bachelor of Technology (Mechanical Engineering)': {
            'B. Tech. (Mech. Eng.)', 'Part-Time: B Tech [Mech Eng]', 'Bachelor of Technology (Mechanical Engineering)',
            '45', '45.0'
        },
        'Bachelor of Computing (Computer Science)': {
            'B.Comp. (Comp.Science)', 'Computer Science', 'Bachelor of Computing (Computer Science)',
            'Bachelor of Computing (Computer S', '30', '30.0'
        },
        'Bachelor of Applied Science': {
            'B.Appl.Sci. (Hons.)', 'B.Appl.Sci.', 'Applied Science [Hons]', 'Applied Science',
            'Bachelor of Applied Science (Hons)', 'Bachelor of Applied Science', 'Bachelor of Applied Science (Hon)',
            '16', '16.0'
        },
        'Bachelor of Computing (Communications and Media)': {
            'B.Comp. (Comm. and Media)', 'Communications and Media', 'Bachelor of Computing (Communications and Media)',
            '27', '27.0'
        },
        'Bachelor of Science (Nursing)': {
            'B.Sc. (Nursing)', 'Nursing [Honours]', 'Nursing', 'Bachelor of Science (Nursing)',
            'Bachelor of Science (Nursing) (Hons)', 'Bachelor of Science (Nursing) (Hon)',
            'Bachelor of Science (Nursing) (Ho', '37', '38', '37.0', '38.0'
        },
        'Bachelor of Dental Surgery': {
            'B.D.S.', 'Dental Surgery', 'Bachelor of Dental Surgery'
        },
        'Bachelor of Engineering (Materials Science and Engineering)': {
            'B.Eng. (Materials Sci. & Eng.)', 'Materials Science and Engineering',
            'Bachelor of Engineering (Materials Science and Engineering)', 'Bachelor of Engineering (Material',
            'Bachelor of Engineering (Materials Science and En', '13', '13.0'
        },
        'Bachelor of Engineering (Environmental Engineering)': {
            'B.Eng. (Environ.)', 'Environmental Engineering', 'Bachelor of Engineering (Environmental Engineering)',
            'Bachelor of Engineering (Environm', 'Bachelor of Engineering (Environmental Engineerin', '11', '11.0'
        },
        'Bachelor of Music': {
            'B.Mus.', 'Music', 'Bachelor of Music', '40', '40.0'
        },
        'Bachelor of Engineering (Engineering Science)': {
            'B.Eng. (Eng. Science)', 'Engineering Science', 'Bachelor of Engineering (Engineering Science)',
            'Bachelor of Engineering (Engineer', '10', '10.0'
        },
        'Bachelor of Arts (Industrial Design)': {
            'B.A. (ID.)', 'Industrial Design', 'Bachelor of Arts (Industrial Design)',
            'Bachelor of Arts (Industrial Desi', '34', '34.0'
        },
        'Bachelor of Computing': {
            'B.Comp.', 'Computing', '26', '26.0'
        },
        'Bachelor of Arts (Architecture)': {
            'B.A. (Arch.)', 'Architecture', 'Follow-Up: Architecture', 'Bachelor of Arts (Architecture)', '33', '33.0'
        },
        'Bachelor of Science (Computational Biology)': {
            'B.Sc. (Comp.Bio.)', 'Computational Biology', 'Bachelor of Science (Computational Biology)',
            'Bachelor of Computing (Computational Biology)', 'Bachelor of Science (Computationa',
            'Bachelor of Computing (Computatio', '19', '28', '19.0', '28.0'
        },
        'Bachelor of Technology (Chemical Engineering)': {
            'B. Tech. (Chem. Eng.)', 'Part-Time: B Tech [Chem Eng]', 'Bachelor of Technology (Chemical Engineering)',
            '41', '41.0'
        },
        'Bachelor of Technology (Manufacturing Engineering)': {
            'B. Tech. (Manufacturing Eng.)', 'Bachelor of Technology (Manufacturing Engineering)', '46', '46.0'
        },
        'Bachelor of Technology (Industrial and Management Engineering)': {
            'Part-Time: B Tech [Industrial and Management Eng]',
            'Bachelor of Technology (Industrial and Management Engineering)',
            'Bachelor of Technology (Industrial and Management Engineerin', '43', '43.0'
        },
        'Bachelor of Environmental Studies': {
            'Bachelor of Environmental Studies', 'Bachelor of Environmental Studies (Biology)',
            'Bachelor of Environmental Studies (Geography)'
        },
        'Bachelor of Science (Business Analytics)': {
            'Bachelor of Science (Business Analytics)', 'Bachelor of Science (Business Ana'
        },
        'Bachelor of Computing (Information Security)': {
            'Bachelor of Computing (Information Security)'
        },
        'Bachelor of Science (Data Science and Analytics)': {
            'Bachelor of Science (Data Science and Analytics)'
        }
    },
    'major': {
        'Accountancy': {'Accountancy', 'ACCOUNTANCY', 'ACCOUNTANCY (HONS)', 'Bus Adm (Accountancy)',
                        'Bus Adm (Accountancy) & Law', 'Bus Adm (Accountancy)(32)', 'Bus Admin (Acc)',
                        'Business Admin (Accountancy)'},
        'Act Studies & Economics': {'ACT ST. & ECONS'},
        'Applied Chemistry': {'Applied Chemistry (Major)', 'APPLIED CHEMISTRY', 'Science (Chemistry/Appl Chem)',
                              'Science (Chemistry/Appl Chem)(2Q)'},
        'Applied Mathematics': {'Applied Mathematics (Major)', 'APPLIED MATHEMATICS', 'APPLIED MATHEMATICS (HONS)'},
        'Architecture': {'ARCHITECTURE', 'ARCHITECTURE (HONS)', 'Architecture', 'Architecture(09)'},
        'Arts & Soc Sci': {'Arts & Soc Sci', 'Arts & Soc Sci(01)'},
        'Bassoon Performance': {'BASSOON PERFORMANCE'},
        'BBA': {'BBA', 'BBA (HONS)', 'Business Admin', 'Business Admin(07)'},
        'Biomedical Engineering': {'BIOENGINEERING', 'BIOMEDICAL ENGINEERING', 'BIOMEDICAL ENGINEERING (HONS)',
                                   'Bioengineering', 'Bioengineering(41)', 'Biomedical Engineering'},
        'Business': {'Business (Major)', 'BUSINESS'},
        'Business Analytics': {'Computing (Business Analytics)'},
        'Cello Performance': {'CELLO PERFORMANCE', 'CELLO PERFORMANCE (HONS)'},
        'Chemical Engineering': {'CHEMICAL ENGINEERING', 'CHEMICAL ENGINEERING (HONS)', 'Chemical Eng',
                                 'Chemical Eng(12)', 'Chemical Engineering'},
        'Chemistry': {'Chemistry (Major)', 'CHEMISTRY', 'CHEMISTRY (HONS)', 'Science (Chemistry)',
                      'Science (Chemistry)(2C)'},
        'Chinese Language': {'Chinese Language (Major)', 'CHINESE LANGUAGE'},
        'Chinese Studies': {'Chinese Studies (Major)', 'CHINESE STUDIES', 'CHINESE STUDIES (HON-CL TRACK)',
                            'CHINESE STUDIES (HONS)'},
        'Civil Engineering': {'CIVIL ENGINEERING', 'CIVIL ENGINEERING (HONS)', 'Civil Eng', 'Civil Eng(42)',
                              'Civil Engineering)'},
        'Clarinet Performance': {'CLARINET PERFORMANCE', 'CLARINET PERFORMANCE (HONS)'},
        'Composition': {'COMPOSITION'},
        'Computational Biology': {'COMPUTATIONAL BIOLOGY', 'COMPUTATIONAL BIOLOGY (HONS)', 'Science (Comp Biology)',
                                  'Science (Comp Biology)(2G)'},
        'Computer Engineering': {'COMPUTER ENGINEERING', 'COMPUTER ENGINEERING-CEG', 'COMPUTER ENGINEERING(HONS)-CEG',
                                 'Computer Eng (FoE)', 'Computer Eng (SoC)', 'Computer Eng', 'Computer Eng (FoE)(19)',
                                 'Computer Eng (SoC)(05)', 'Computer Eng(48)', 'Computer engineering', 'Comp Eng (FoE)',
                                 'Computer Engineering (FOE & SOC)'},
        'Computer Science': {'COMPUTER SCIENCE', 'COMPUTER SCIENCE (HONS)', 'Computing (CS)',
                             'Computing (Computer Science)'},
        'Computing': {'Computing', 'Computing(13)'},
        'Communications & Media': {'COMMUNICATIONS & MEDIA', 'COMMUNICATIONS & MEDIA (HONS)'},
        'Communications & New Media': {'Comms & New Media (Major)', 'Comms and New Media (Major)', 'COMMS & NEW MEDIA',
                                       'COMMS & NEW MEDIA (HONS)'},
        'Dental Surgery': {'Dental Surgery (Major)', 'DENTISTRY', 'Dentistry', 'Dentistry(04)'},
        'Double Bass Performance': {'DOUBLE BASS PERFORMANCE', 'DOUBLE BASS PERFORMANCE (HONS)'},
        'E-Commerce': {'ELECTRONIC COMMERCE', 'ELECTRONIC COMMERCE (HONS)'},
        'Economics': {'Economics (Major)', 'ECONOMICS', 'ECONOMICS (HONS)', 'Economics and Law'},
        'Electrical Engineering': {'ELECTRICAL ENGINEERING', 'ELECTRICAL ENGINEERING (HONS)', 'Electrical Eng',
                                   'Electrical Eng(43)', 'Electrical Engineering'},
        'Electronics Engineering': {'ELECTRONICS ENGINEERING', 'ELECTRONICS ENG (HONS)'},
        'Engineering': {'Engineering', 'Engineering (14)'},
        'Engineering Science': {'ENGINEERING SCIENCE', 'ENGINEERING SCIENCE (HONS)', 'Engineering Science',
                                'Engineering Science (47)'},
        'English Language': {'English Language (Major)', 'ENGLISH LANGUAGE', 'ENGLISH LANGUAGE (HONS)'},
        'English Literature': {'English Lit (Major)', 'ENGLISH LIT', 'ENGLISH LIT (HONS)'},
        'Environmental Engineering': {'ENVIRONMENTAL ENG', 'ENVIRONMENTAL ENG (HONS)', 'Environmental Eng',
                                      'Environmental Eng(18)', 'Environ Eng'},
        'Environmental Studies': {'ENVIRONMENTAL STUDIES-BIO HONS', 'ENVIRONMENTAL STUDIES-GEO HONS',
                                  'Environmental Studies'},
        'European Studies': {'European Studies (Major)', 'EUROPEAN STUDIES', 'EUROPEAN STUDIES (HONS)'},
        'Flute Performance': {'FLUTE PERFORMANCE', 'FLUTE PERFORMANCE (HONS)'},
        'Food Science & Tech': {'Food Science & Tech (Major)', 'FOOD SCIENCE & TECH', 'Food Science and Tech (Major)',
                                'FOOD SCIENCE & TECH (HONS)', 'Food Sci & Tech (Major)',
                                'Science (Food Sci & Techn\'gy)', 'Appl Sci (Food Sci & Techn\'gy)(2F)',
                                'Appl Sci (Food Sci & Techn\'gy)'},
        'French Horn Performance': {'FRENCH HORN PERFORMANCE', 'FRENCH HORN PERFORMANCE (HONS)'},
        'Geography': {'Geography (Major)', 'GEOGRAPHY', 'GEOGRAPHY (HONS)'},
        'Global Studies': {'Global Studies (Major)', 'GLOBAL STUDIES (HONS)'},
        'Harp Performance': {'HARP PERFORMANCE', 'HARP PERFORMANCE (HONS)'},
        'History': {'History (Major)', 'HISTORY', 'HISTORY (HONS)'},
        'Industrial & Management Engineering': {'INDUSTRIAL AND MANAGEMENT ENGINEERING', 'INDUSTRIAL & MGT ENG (HONS)'},
        'Industrial & System Engineering': {'INDUSTRIAL & SYS ENG', 'INDUSTRIAL AND SYSTEMS ENGINEERING',
                                            'INDUSTRIAL & SYS ENG (HONS)', 'Indtl & Systems Eng',
                                            'Indtl & Systemse Eng(45)', 'Industrial & Systems Engineering',
                                            'Indtl & Sys Engineering'},
        'Industrial Design': {'INDUSTRIAL DESIGN', 'INDUSTRIAL DESIGN (HONS)', 'Industrial Design',
                              'Industrial Design(20)'},
        'Information Systems': {'INFORMATION SYSTEMS', 'INFORMATION SYSTEMS (HONS)', 'Computing (IS)(57)',
                                'Computing (Information Systems)', 'Computing (Info Systems)'},
        'Japanese Studies': {'Japanese Studies (Major)', 'JAPANESE STUDIES', 'JAPANESE STUDIES (HONS)'},
        'Law': {'LAW', 'LAW (HONS) - 4 YEAR', 'LAW (HONS) - 3 YEAR', 'Law', 'Graduate Law Programme', 'Law(06)',
                'Law (Graduate Programme)'},
        'Life Sciences': {'Life Scences (Major)', 'LIFE SCIENCES', 'LIFE SCIENCES (HONS)', 'Science (Life Sciences)',
                          'Science (Life Sciences)(2L)'},
        'Malay Studies': {'Malay Studies (Major)', 'MALAY STUDIES', 'MALAY STUDIES (HONS)'},
        'Materials Science & Engineering': {'MATERIALS SC & ENG', 'MATERIALS SCIENCE AND ENGINEERING',
                                            'MATERIALS SC & ENG (HONS)', 'Materials Sci & Eng',
                                            'Materials Sci & Eng(46)', 'Materials Sci & Engineering'},
        'Mathematics': {'Mathematics (Major)', 'MATHEMATICS', 'MATHEMATICS (HONS)'},
        'Mechanical Engineering': {'MECHANICAL ENGINEERING', 'MECHANICAL ENGINEERING (HONS)', 'Mechanical Eng',
                                   'Mechanical Eng(44)', 'Mechanical Engineering'},
        'Medicine': {'MEDICINE', 'Medicine (Major)', 'Medicine', 'Medicine(03)'},
        'Music': {'Music', 'Check with YSTCM'},
        'Nursing': {'Nursing (Major)', 'NURSING', 'NURSING (HONS)', 'Nursing (Major) PCP', 'Nursing', 'Nursing(51)'},
        'Oboe Performance': {'OBOE PERFORMANCE (HONS)'},
        'Percussion Performance': {'PERCUSSION PERFORMANCE', 'PERCUSSION PERFORMANCE (HONS)'},
        'Piano Performance': {'PIANO PERFORMANCE', 'PIANO PERFORMANCE (HONS)'},
        'Political Science': {'Political Science (Major)', 'POLITICAL SCIENCE', 'POLITICAL SCIENCE (HONS)'},
        'Pharmacy': {'PHARMACY', 'PHARMACY (HONS)', 'Pharmacy', 'Pharmacy(15)'},
        'Philosophy': {'Philosophy (Major)', 'PHILOSOPHY', 'PHILOSOPHY (HONS)'},
        'Physics': {'Physics (Major)', 'PHYSICS', 'PHYSICS (HONS)', 'Science (Physics)', 'Science (Physics)(2P)'},
        'Project & Facilities Management': {'PROJECT & FAC. MGT', 'PROJECT & FAC. MGT (HONS)',
                                            'Project & Facilities Mgt', 'Project & Facilities Mgt(30)'},
        'Psychology': {'Psychology (Major)', 'PSYCHOLOGY', 'PSYCHOLOGY (HONS)'},
        'Quantitative Finance': {'Quantitative Finance (Major)', 'QUANTITATIVE FINANCE', 'QUANTITATIVE FINANCE (HONS)',
                                 'Quantitative Finance (Major'},
        'Real Estate': {'REAL ESTATE', 'REAL ESTATE (HONS)', 'Real Estate', 'Bldg/Real Est', 'Real Estate(31)'},
        'Recording Arts & Science': {'RECORDING ARTS & SCI.', 'RECORDING ARTS & SCI. (HONS)'},
        'S.E. Asian Studies': {'S.E. Asian Studies (Major)', 'S.E. ASIAN STUDIES', 'S.E. ASIAN STUDIES (HONS)'},
        'Science': {'Science', 'Science(02)'},
        'Social Work': {'Social Work (Major)', 'SOCIAL WORK', 'SOCIAL WORK (HONS)'},
        'Sociology': {'Sociology (Major)', 'SOCIOLOGY', 'SOCIOLOGY (HONS)'},
        'South Asian Studies': {'SOUTH ASIAN STUD', 'South Asian Stud (Major)'},
        'Statistics': {'Statistics (Major)', 'STATISTICS', 'STATISTICS (HONS)'},
        'Technology': {'Technology (Major)'},
        'Theatre Studies': {'Theatre Studies (Major)', 'THEATRE STUDIES', 'THEATRE STUDIES (HONS)'},
        'Trombone Performance': {'TROMBONE PERFORMANCE', 'TROMBONE PERFORMANCE (HONS)'},
        'Trumpet Performance': {'TRUMPET PERFORMANCE', 'TRUMPET PERFORMANCE (HONS)'},
        'Tuba Performance': {'TUBA PERFORMANCE'},
        'Viola Performance': {'VIOLA PERFORMANCE', 'VIOLA PERFORMANCE (HONS)'},
        'Violin Performance': {'VIOLIN PERFORMANCE', 'VIOLIN PERFORMANCE (HONS)'},
        'Voice': {'VOICE', 'VOICE (HONS)'},
    },
    'usp': {
        'Y': {'1', '3', '4', '1.0', '3.0', '4.0', 'Yes'},
        'N': {'2', '2.0', '.', 'No'}
    },
    'faculty': {
        'Faculty of Engineering': {'Faculty of Engineering', '3', '3.0', 'FACULTY OF ENGINEERING'},
        'Faculty of Arts & Social Sciences': {'Faculty of Arts & Social Sci', '1', '1.0', 'Arts & Social Sciences',
                                              'Faculty of Arts & Social Science', 'Faculty of Arts & Social Sciences',
                                              'FACULTY OF ARTS & SOCIAL SCI'},
        'Faculty of Science': {'Faculty of Science', '5', '5.0', 'FACULTY OF SCIENCE'},
        'NUS Business School': {'NUS Business School', '6', '6.0', 'NUS BUSINESS SCHOOL'},
        'Faculty of Law': {'Faculty of Law', '4', '4.0', 'FACULTY OF LAW'},
        'School of Cont & Lifelong Edun': {'School of Cont & Lifelong Edun', '8', '8.0'},
        'School of Computing': {'School of Computing', '7', '7.0', 'SCHOOL OF COMPUTING'},
        'YLL School of Medicine': {'YLL School of Medicine', '9', '9.0', 'YONG LOO LIN SCHOOL (MEDICINE)',
                                   'Yong Loo Lin School (Medicine)', 'Yong LOo Lin Sch of Medicine'},
        'School of Design & Environment': {'School of Design & Environment', '2', '2.0',
                                           'School of Design and Environment', 'SCHOOL OF DESIGN & ENVIRONMENT'},
        'Faculty of Dentistry': {'Faculty of Dentistry', 'FACULTY OF DENTISTRY'},
        'YST Conservatory of Music': {'YST Conservatory of Music', '10', '10.0', 'Yong Siew Toh Conservatory of Music',
                                      'YST CONSERVATORY OF MUSIC'},
        'Multi Disciplinary Programme': {'Multi Disciplinary Programme', 'MULTI DISCIPLINARY PROGRAMME',
                                         'Multidisciplinary Programme'},
        'Yale-NUS College': {'Yale-NUS College'}
    },
    'noc': {
        'Y': {'1', '1.0', 'Y', 'NUS Overseas College', 'Yes'},
        'N': {'2', '2.0', 'Not NOC', 'N', 'No'}
    },
    'segment_als': {
        'Full-time': {'Full-time', 'FT'},
        'Part-time': {'Part-Time', 'Part-time', 'PT'},
        'Follow-up': {'Follow-up', 'FU'}
    },
    'hon': {
        'Y': {'Honours Degree', 'Y', 'HON', 'Hon', 'Honours'},
        'N': {'General Degree', 'N', 'NO HON', 'No Hon', 'Non Honours'}
    },
    'class_of_hons': {
        '2nd Class Honours (Upper)': {'3', '3.0', '2nd Class Honours (Upper)', '2nd Class (Upper Division)',
                                      'Second class (Upper division)', '2A', '2nd Class Honours (Upper Division)'},
        '2nd Class Honours (Lower)': {'4', '4.0', '2nd Class Honours (Lower)', '2nd Class (Lower Division)',
                                      'Second class (Lower division)', '2B', '2nd Class Honours (Lower Division)'},

        'Pass': {'7', '7.0', 'Pass', 'P'},
        'Merit': {'8', '8.0', 'Pass with Merit', 'Merit', 'PM'},
        '1st Class Honours': {'2', '2.0', '1st Class Honours', 'First class', 'A1'},
        '3rd Class Honours': {'5', '5.0', '3rd Class Honours', 'Third class', 'A3'},
        'Honours': {'.', 'Honours', 'Not specified', 'Honours - Not specified', 'H'},
        'Honours (Merit)': {'HM', 'Honours (Merit)'},
        'Honours (Distinction)': {'HD', 'Honours (Distinction)'},
        'Honours (Highest Distinction)': {'HHD', 'Honours (Highest Distinction)'}
    },
    'progtype': {
        'NORMAL': {'NORMAL', 'Normal'},
        'CDP': {'CDP'},
        'CDM': {'CDM'},
        'DDP': {'DDP'},
        'JDP': {'JDP'}
    },
    'is_first_choice': {
        'Y': {'Y', 'Yes', 'Current course was my first choice of study',
              'Currrent course was my first choice of study'},
        'N': {'N', 'No', 'Current course was not the first choice of study'},
        'Not Sure': {'I did not have an idea of what I wanted to pursue when applying to university.'}
    },
    'professional_degree': {
        'Professional Degree': {'Professional', 'Professional Degree'},
        'Non-Professional Degree': {'Non-Professional', 'Non-Professional Degree', 'Non Professional Degree'}
    },
    'residency': {
        'Singapore Citizen': {'Singapore Citizen', '1', '1.0', 'Singaporean', 'Singapore Cit'},
        'Permanent Resident': {'Permanent Resident', 'Singapore Permanent Resident', '2', '2.0', 'Singapore PR',
                               'SingaporeanSingapore PR', 'Singapore Per'},
        'International': {'Others', 'International', '3', '3.0', 'SingaporeanOthers', '4', '4.0', '5', '5.0', '7',
                          '7.0', 'Singaporean6'}
    },
    'fin_src': {
        'Self-financing': {'Self-financing (E.g. Family resources)', '1', '1.0'},
        'Tuition Fee Loan': {'Tuition Fee Loan', '2', '2.0'},
        'Ministry / Statutory Board Scholarships': {'Ministry / Statutory Board Scholarships', '6', '6.0'},
        'University Scholarships': {
            'University Scholarships (e.g. NUS Global Merit Scholarship)', '9', '9.0',
            'University Scholarships (e.g. NUS Global Merit Scholarship, Full Scholarship offered by NUS)'
        },
        'Private Sector Scholarships': {
            'Private Sector Scholarships (e.g. offered by DBS, SIA-NOL, S', '4', '4.0',
            'Other Private Sector Scholarships (e.g. private companies an',
            'Other Private Sector Scholarships (e.g. private companies and clan etc.) (Please specify)'
        },
        'University Administered Financial Assistance': {
            'University Administered Financial Assistance (e.g. Study Loa', '3', '3.0',
            'University Administered Financial Assistance (e.g. Study Loan, Bursaries)'
        },
        'Community Organisation Scholarship': {
            'Community Organisation Scholarships (E.g. CDC, Mendaki, SIND', '7', '7.0',
            'Community Organisation Scholarships (E.g. CDC, Mendaki, SINDA) (Please specify)'
        },
        'Public Service Commission (PSC) Scholarships': {'Public Service Commission (PSC) Scholarships', '8', '8.0'},
        'Other Financial Assistance': {
            'Other Financial Assistance', 'Other Scholarships', '10', '10.0',
            'Other Financial Assistance (Please specify)'
        },
        'Refused': {'Refused', '5', '5.0', 'Do not wish to disclose'},
        'PRC Undergraduate Scholarship': {'PRC Undergraduate Scholarship'},
        'ASEAN Undergraduate Scholarship': {'ASEAN Undergraduate Scholarship'},
        'Sembcorp Scholarship': {'Sembcorp Scholarship'},
        'SIA-NOL Scholarship': {'SIA-NOL Scholarship'},
        'Science & Technology Undergraduate Scholarship': {
            'Science &amp; Technology Undergraduate Scholarship', 'Science & Technology Undergraduate Scholarship'
        }
    },
    'new_citizen': {
        'Not Applicable. Born Singaporean.': {'Not Applicable. I am a Singaporean at birth.', '14', '14.0'},
        'Not Applicable. Born PR.': {'Not Applicable. I am a Permanent Resident at birth.', '13', '13.0'},
        'Before 2000': {'Before 2000', '1', '1.0'},
    } | {
        str(1998+x): {str(1998+x), str(x), f'{x}.0'}
        for x in range(2, 18)
    },
    'activity_status': {
        'Full-Time Permanent': {'Working Full-Time in a Permanent job', '1', '1.0'},
        'Practical Training': {'Currently undergoing practical training related to the cours', '2', '2.0',
                               'Currently undergoing practical training related to the course of study',
                               'Currently undergoing practical training related to the course',
                               'Currently undergoing practical training related to the course o',
                               'Currently undergoing practical train'},
        'Actively Looking For': {'Not working but actively looking and available for work', '3', '3.0',
                                 'Not working but actively looking and'},
        'Further Studies': {'Currently pursuing / preparing to commence further studies,', '4', '4.0',
                            'Reading for Higher Degree', 'Other further studies not in list',
                            'k) Other further studies not in list', 'a) Reading for Higher Degree',
                            'Currently pursuing/ preparing to commence further studies, excluding NIE',
                            'Currently pursuing \\ preparing to commence further studies, exc',
                            'Currently pursuing \\ preparing to commence further studies, excluding NIE.',
                            'Currently pursuing\\ preparing to com',
                            'Currently pursuing\\ preparing to commence further studies, excluding NIE.',
                            'Currently pursuing\\ preparing to commence further studies, excluding NIE. Note: General '
                            'Education Officer enrolled in NI'},
        'Full-Time Temporary': {'Working Full-Time in a Temporary job'},
        'Offered': {'Accepted job offer and will start later', 'Accepted job offer and will start la'},
        'Idle': {'Not working and not looking for a job', 'Not working and not looking for a jo'},
        'Part-Time Temporary': {'Working Part-Time in a Temporary job'},
        'Business Venture': {'Taking steps to start a business venture'},
        'Part-Time Permanent': {'Working Part-Time in a Permanent job'},
        'Medical Graduate': {'Medical graduate serving housemanship', 'Medical graduate serving housemanshi'},
        'Law Graduate': {'Law graduate doing practical law course (PLC) / Pupilage or',
                         'Law graduate doing practical law course (PLC) / Pupilage or reading in chambers',
                         'Law graduate doing practical law course (PLC) / Pupilage',
                         'Law graduate doing practical law course (PLC)\\ Pupilage or reading in chambers',
                         'Law graduate doing practical law course (PLC) \\ Pupilage or rea',
                         'Law graduate doing practical law course (PLC) \\ Pupilage or reading in chambers.',
                         'Law graduate doing practical law course (PLC)\\ Pupilage or reading in cha',
                         'Law graduate doing practical law cou',
                         'Law graduate doing practical law course (PLC)\\ Pupilage or reading in chambers.',
                         'Law graduate doing practical law course (PLC)\\ Pupilage or reading in chambers. [Applicable '
                         'for Law graduates from Class'},
        'Pharmacy Graduate': {'Pharmacy graduate serving pupilage'},
        'Architecture Graduate': {'B.Arch graduate undergoing year-out practical training'},
        'Master by Coursework': {'Master’s Degree by coursework'},
        'Master by Research': {'Master’s Degree by research'},
        'PhD': {'PhD'},
        'Graduate Diploma': {'Graduate Diploma'},
        'MBA': {'MBA'},
        'Second Undergraduate': {'Second Undergraduate Degree'},
        'PGDE': {'Pursuing Post Graduate Diploma in Education (PGDE) without s',
                 'Pursuing Post Graduate Diploma in Education (PGDE) without service bond from MOE'},
        'Others': {'Others'},
        'CFA': {'Professional Qualification – CFA equivalent'},
        'CFP': {'Professional Qualification – CFP equivalent'},
        'Freelance': {'Working on a Freelance basis'}
    },
    'not_ft_perm_reason': {
        'Unable': {'2', '2.0', 'Tried to but unable to find Full-Time Permanent Job',
                   'Tried to but unable to obtain Full-Time Permanent Job offer'},
        'Try out': {'3', '3.0', 'To try out and see if this field / job is suitable for me',
                    'To try out and see if this field/ job is suitable for me',
                    'To try out and see if this field \\ job is suitable for me',
                    'To try out and see if this field\\ job is suitable for me'},
        'Further studies': {'1', '1.0', 'Currently pursuing / preparing to commence further studies',
                            'Currently pursuing/ preparing to commence further studies',
                            'Currently pursuing \\ preparing to commence further studies',
                            'Currently pursuing\\ preparing to commence further studies'},
        'Personal': {'4', '4.0', 'Personal choice'},
        'Others': {'5', '5.0', 'Others', 'Others (please specify)', 'Others, please specify:'},
        'Business venture': {'Taking steps to start a business venture'},
        'Waiting': {'Waiting for job offers / looking for jobs',
                    'Working part time while waiting for full time permanent job to commence',
                    'To kill time / Waiting for new job / holiday etc', 'Accepted job offer and will start work later',
                    'Waiting for confirmation of job offer'},
        'Other commitments': {'Pursuing other commitments at the same time'},
        'Refused': {'Refused to disclose', 'Refused'},
        'Potential conversion': {'Waiting for conversion into full-time position', 'Waiting for conversion'},
        'Gain experience': {'To gain experience in this field', 'Gain exposure / experience / skills'},
        'Training': {'Pre-registration training  / Pupilage / Internship', 'Training period / Pupillage',
                     'Pre registration training / internship'},
        'Contract': {'Contract / freelance basis', 'On contract work'},
        'Income': {'For income', 'To earn some income while searching for my ideal full-time permanent job'},
        'Re-skilling': {'Undergoing external re-skilling programmes (Professional conversion programme by WSG)'},
        'Flexibility': {'Preference for shorter and \\ or flexible work hours'},
        'Nature': {'Nature of occupation I choose to enter offers mostly freelance \\ temp \\ part-time work',
                   'Nature of occupation I choose to enter offers mostly freelance \\ temp \\'},
        'Nil': {'.', 'Non Applicable based on skip pattern', 'Not applicable based on skipping pattern'}
    },
    'employment_status': {
        'Employee': {'1', '1.0', 'Employee'},
        'Self-employed': {'2', '2.0', 'Self-employed'},
        'Nil': {'.', 'Not Applicable based on skip pattern', 'Not applicable based on skipping pattern'}
    },
    'company_type': {
        'Private': {'Private firm - Multinational Company (MNC)', '1', '1.0',
                    'Private firm - Government-linked Company (GLC)', '2', '2.0',
                    'Private firm – Small-Medium Enterprise (SME)', '3', '3.0',
                    'Private firm (including government-linked companies)',
                    '(For employee only) Private firm (including government-linked companies)'},
        'Government': {'Government', '6', '6.0', '(For employee only) Government',
                       '(For employee only) Government (including Ministries and State Organisations)',
                       'Government (including Ministries and State Organisations)'},
        'Statutory board': {'Statutory board', '4', '4.0', '(For employee only) Statutory board'},
        'Voluntary organisation': {'Voluntary organisation', '5', '5.0', '(For employee only) Voluntary organisation'},
        'Own': {'8', '8.0'},
        'Entrepreneur': {'Own business in a <u>non-technical</u> field (entrepreneur)',
                         'Own business in a non-technical field (entrepreneur)',
                         '(For self-employed only) Own business in a non-technical field (entrepreneur)'},
        'Technopreneur': {'Own business in a <u>technical</u> field (technopreneur)',
                          'Own business in a technical field (technopreneur)',
                          '(For self-employed only) Own business in a technical field (technopreneur)'},
        'Family': {'Own family business', '(For both) Own family business'},
        'Others': {'Others <i>(Please specify)</i>', '99', '10.0', 'Others', 'Others, please specify:',
                   'Others, please specify'},
        'Refused': {'Refused to disclose'},
        'Freelance': {'Self-employed / Freelancer / Insurance / property agents'},
        'Reiligious': {'Religious Organisation'},
        'Nil': {'.', 'Non Applicable based on skip pattern', 'Not applicable based on skipping pattern'}
    },
    'industry': {
        'Nil': {'.', 'Non Applicable based on skip pattern', 'Not applicable based on skipping pattern'},
        'Public Admin & Defense': {
            'Public Admin (Ministries, Stat Boards except Defence)', '22', '22.0', '15', '32',
            'Public Administration and Defence', '23', 'Defence & Security - MINDEF, Police, Civil Defence, etc',
            'Security & Investigative Activities', 'Investigation & Security'
        },
        'Healthcare': {
            'Healthcare', '23', '23.0', 'Healthcare (Includes hospitals, medical, dental activities &',
            'Healthcare (Includes hospitals, medical, dental activities & pharmacies)'
        },
        'Information and Communications': {
            'Information & Communications', 'Publishing', 'Communications & Media',
            'Information & Communication (Includes publishing, motion pic', '16', '16.0',
            'Information & Communication (Includes publishing, motion pictures & video production, radio & television '
            'broadcasting ac', 'Information & Communication',
            'Information & Communication (Includes publishing, motion picture & video production, radio & television '
            'broadcasting act'
        },
        'IT': {'IT Services (Computer Programming, Data, Web)', 'IT & Information services'},
        'Financial': {
            'Financial and Insurance', 'Financial Services & Insurance', '2', '2.0', '21', '21.0', '30', '30.0', '27',
            '27.0'
        },
        'Legal, Accounting and Auditing': {'Legal, Accounting and Auditing', '26', '26.0'},
        'Electronic Products': {
            'Manufacturing (Semicon, Communications, Electronics)', 'Manufacturing - Electronic Products',
            'Manufacturing (Consumer Electronics)', 'Electronic Products (includes semiconductor, communications',
            'Electrical Products', 'Electronic Products (includes semiconductor, communications equipment, etc.)',
            'Electronic Products (includes semiconductor, communications equipment, computing & data processing '
            'equipment, TV & radio', 'Electronic Products'
        },
        'Gasoline': {
            'Manufacturing (Refined Petroleum Pdts)', '10', '31', '10.0', '31.0',
            'Oilfield and gasfield machinery and equipment manufacturing',
            'Petroleum, mining and prospecting services (Including offsho',
            'Petroleum, mining and prospecting services (Including offshore exploration services)',
            'Petroleum, mining and prospecting services', 'Oilfield and gasfield machinery and equipment manufac'
        },
        'Aerospace': {'Aerospace'},
        'Offshore': {
            'Marine and offshore engineering', 'Maritime/Shipping', 'Maritime/ Shipping', 'Maritime\\Shipping',
            'Maritime\\ Shipping'
        },
        'Pharmaceutical Products': {
            'Pharmaceutical & Biological Products manufacturing', 'Pharmaceutical &  Biological Products manufacturing',
            'Manufacturing (Pharmaceuticals & Biological Pdts)'
        },
        'Chemical': {
            'Manufacturing - Chemical incl Petrochemical', 'Chemical manufacturing', '1', '13', '1.0', '13.0',
            'Manufacturing (Chemicals, Plastics & Rubber)', '4', '4.0'
        },
        'Transport Equipment': {
            'Transport Equipment', 'Manufacturing (Non-motor vehicle Transport)',
            'Manufacturing (Motor vehicles, Trailers, Semi-trailers)', 'Transport Equipment (excluding Aerospace)'
        },
        'Medical Instruments': {'Medical & precision instruments', 'Medical and Precision Instruments'},
        'Machinery': {'Manufacturing (Machinery & Equipment)', 'Machinery and Equipment'},
        'Metal': {
            'Manufacturing (Metallic Pdts except Machinery/Equipment)', 'Manufacturing (Basic Metals)',
            'Fabricated Metal Products'
        },
        'Light': {'Light and other manufacturing', 'Manufacturing - Light & Others'},
        'Manufacturing': {
            'Other manufacturing (includes wood, glass products & furnitu', '39', '39.0', 'Other manufacturing',
            'Other engineering manufacturing', 'Other manufacturing (includes wood, glass products & furniture)',
            'Other manufacturing (includes wood, glass products &amp; fur'
        },
        'Engineering': {
            'Other engineering activities (general building, process plan',  'Manufacturing - Engineering', '7.0',
            'Other engineering activities (general building, process plant, industrial plant, ' 
            'infrastructure engineering services)', '9', '24', '9.0', '24.0', 'Other engineering activities', '7', '12',
            'Industrial design activities', 'Other engineering services activities', '11', '11.0', '17', '17.0', '12.0',
            '29', '20.0', '33', '33.0', '3', '3.0', '25', '25.0', '38', '38.0', '6', '6.0', '32.0', '29.0'
        },
        'Printing': {'Paper Products & Printing', 'Paper Products &amp; Printing'},
        'Education': {
            'Education', 'Education (Cultural - Non-Academic)', '20', '20.0', '18', '18.0', '19', '19.0',
            'Educational Support Services (Pte Tutors, etc)'
        },
        'Scientific R&D': {
            'Scientific R&D (Natural Sciences & Engineering)', '36', '36.0', 'Scientific Research & Development',
            'Scientific Research & Development (Includes R&D on Natural S',
            'Professional, Scientific and Technical Activities necs', 'Life Sciences',
            'Other Professional, Scientific & Technical activities (Inclu',
            'Scientific Research & Development (Includes R&D on Natural Sciences & Engineering, Social Sciences & '
            'Humanities)', 'Other Professional, Scientific & Technical activities',
            'Other Professional, Scientific & Technical activities (Includes Specialized Design Activities, '
            'Photographic Activities)', 'Other Professional, Scientific & Technical activit',
            'Other Professional, Scientific & Technical activities (Includes Specialized Design Activities (excluding '
            'Industrial Desi', '34', '34.0'
        },
        'Administrative': {
            'Administrative and Support Services Activities (Includes Ren', 'Administrative & Support Services',
            'Administrative and Support Services Activities',
            'Administrative and Support Services Activities (Includes Rental & Leasing, Employment Agencies, Travel '
            'Agencies, Investi',
            'Administrative and Support Services Activities (Includes Rental & Leasing, Employment Agencies, Travel '
            'Agencies etc.)',
            'Administrative and Support Services Activities (Includes Rental & Leasing, Employment Agencies, Travel '
            'Agencies, Buildin', '40', '40.0', '32.0'
        },
        'Food & Beverage': {
            'F&B Services', 'Food & Beverage ( Includes Restaurants, Bars, Food Caterers,', 'Food, Beverages & Tobacco',
            'Manufacturing (F&B, Tobacco, Textiles, Apparel, Wood, Paper)',
            'Food & Beverages (Includes Restaurants, Bars, Food Caterers, Hawkers)',
            'Food & Beverages (Includes Restaurants, Bars, Food Caterers,', 'Food & Beverage',
            'Food & Beverage ( Includes Restaurants, Bars, Food Caterers, Hawkers)',
            'Food & Beverage (Includes Restaurants, Bars, Food Caterers, Hawkers)'
        },
        'Personal Services': {
            'Personal & Other Service Activities (Includes  Membership Or',
            'Personal Svcs (Laundries, Hair & Beauty, Weddings, Funerals)',
            'Membership Orgainizations (Clubs, Unions, Religious, Politic',
            'Employer & Professional Membership Organisations',
            'Personal & Other Service Activities (Includes  Membership Organisations, Repair of Computers, Personal & '
            'Household Goods', 'Personal & Other Service Activities'},
        'Accommodation': {
            'Accomodation/Hospitality (Hotels, Hostels, Serviced Apts)', 'Accommodation (Includes Hotels)',
            'Tourism & Hospitality', '24', '24.0', 'Accommodation'
        },
        'Society & Community': {
            'Society & Community (Includes child-care services, social se', '37', '37.0',
            'Social & Community Services (without Accomodation)', 'Social & Community',
            'Society & Community (Includes child-care services, social services for families / elderly, activities '
            'related to communi', 'Society & Community',
            'Society & Community (Includes child-care services, social services for families\\ elderly, activities '
            'related to communit'
        },
        'Architectural': {
            'Architectural & Engineering; Technical Testing', '8', '8.0',
            'Architectural  (Includes architectural services, landscape d',
            'Architectural  (Includes architectural services, landscape design & architecture), Engineering, Land '
            'Surveying & Technic', 'Architectural', 'Architectural, Engineering, Land Surveying & Technical Services)',
            'Architectural, Engineering, Land Surveying & Technical Services',
            'Architectural (Includes architectural services, landscape design & architecture), Engineering, Land '
            'Surveying & Technica'},
        'Retail Trade': {'Retail Trade'},
        'Wholesale Trade': {'Wholesale Trade'},
        'Extra-Territorial Organisations': {
            'Extra-Territorial Organisations', 'Activities of Extra-territorial Organisation and Bodies'
        },
        'Entertainment': {
            'Creative Arts & Entertainment', 'Arts, Entertainment, Library, Sports & Recreation',
            'Arts, Entertainment and Recreation (Includes sports activiti',
            'Arts, Entertainment and Recreation (Includes sports activities, libraries)',
            'Arts, Entertainment and Recreation'
        },
        'Consultancy': {
            'Business & Management Consultancy', 'Head Offices & Management Consultancy Activities', '2', '2.0',
            'Business and Management Consultancy (Includes Head Offices &', 'Business and Management Consultancy',
            'Business and Management Consultancy (Includes Head Offices & business representative offices)'
        },
        'Advertising & Market': {'Advertising Services & Market Research', 'Advertising services and Market Research'},
        'Others': {
            'Activities not adequately defined', 'Others (Includes Agriculture - Growing of Crops/Nursery Prod',
            'Others (Includes Agriculture - Growing of Crops/Nursery Prod', 'Others',
            'Others (Includes Agriculture - Growing of Crops/Nursery Products & Horticulture, Fishing, Mining and '
            'Quarrying, Electric', 'Others please specify', 'Others, Please specify', '5.0', '15.0'
        },
        'Resources': {
            'Water Supply & Waste Management', 'Electricity, Gas & Airconditioning Supply',
            'Solar, wind, water treatment', 'Electricity, Gas And Air-Conditioning Supply'
        },
        'Transport & Storage': {
            'Transportation &amp; Storage (Includes land transport,  air', 'Postal & Courier Svcs',
            'Transport, Logistics, Storage - Land\\Air\\Water, allied svcs', 'Air Transport', 'Water Transport',
            'Logistics, Warehousing & Storage', 'Logistics and Supply Chain Management',
            'Transportation & Storage (Includes land transport,  air tranport, water transport, warehousing & support '
            'activities for', '28', '28.0','Transportation & Storage', 'Other transportation & storage',
            'Transportation & Storage (Includes land transport,  air tran', 'Land Transport',
            'Other transportation & Storage', 'Other Transportation & Storage',
            'Water Transport (including Maritime\\Shipping)'
        },
        'Refuse': {'Refuse to disclose', 'Refused'},
        'Cleaning': {'Cleaning'},
        'Employment & Recruitment': {'Employment & Recruitment'},
        'Telecommunications': {'Telecommunications'},
        'Real Estate': {
            'Real Estate Services', 'Real Estate', 'Real Estate (Sales, Rental & Leasing)', '35', '35.0',
            'Real Estate(Sales, Rental & Leasing)', '14', '14.0'
        },
        'Construction': {'Construction', 'Construction & Civil Enginerering'},
        'Textile': {'Textile & Wearing Apparel', 'Textile & Wearing Apparels'}
    },
    'gap_grad_search': {
        '1 to 4 weeks after graduation': {'1 to 4 weeks after graduation', 'Within month of graduation'} |
                                         {str(i) for i in range(5)},
        '5 to 8 weeks after graduation': {'5 to 8 weeks after graduation',  'Between 1 and 2 months after graduation'} |
                                         {str(i) for i in range(5, 9)},
        '9 to 12 weeks after graduation': {'9 to 12 weeks after graduation'} | {str(i) for i in range(9, 13)},
        '13 to 16 weeks after graduation': {'13 to 16 weeks after graduation',
                                            'Between 3 and 4 months after graduation'} |
                                           {str(i) for i in range(13, 17)},
        '17 to 20 weeks after graduation': {'17 to 20 weeks after graduation'} | {str(i) for i in range(17, 21)},
        '21 to 24 weeks after graduation': {'21 to 24 weeks after graduation',
                                            'Between 5 and 6 months after graduation'} |
                                           {str(i) for i in range(21, 25)},
        '25 to 28 weeks after graduation': {'25 to 28 weeks after graduation'} | {str(i) for i in range(25, 29)},
        'More than 28 weeks after graduation': {'More than 28 weeks after graduation', '29', '30',
                                                'More than 6 months after graduation'},
        '1 to 8 weeks before graduation': {'1 to 8 weeks before graduation', '1 month before graduation'} | {str(i) for i in range(-8, 0)},
        '9 to 16 weeks before graduation': {'9 to 16 weeks before graduation',
                                            'Between 2 and 3 months before graduation'} |
                                           {str(i) for i in range(-16, -8)},
        '17 to 24 weeks before graduation': {'17 to 24 weeks before graduation',
                                             'Between 4 and 6 months before graduation'} |
                                            {str(i) for i in range(-24, -16)},
        'More than 24 weeks before graduation': {'More than 24 weeks before graduation',
                                                 'At least 6 months before graduation'} |
                                                {str(i) for i in range(-30, -24)}
    },
    'offer_count': {
        'One': {'One', '2.0', '5'},
        'Two': {'Two', '3.0', '6'},
        'Three': {'Three', '4.0', '7'},
        'Four': {'Four', '5.0', '8'},
        'Five': {'Five', '6.0', '9'},
        'Six or more': {'Six or more', '7.0', '10'},
        'Nil': {'.', 'Non Applicable based on skip pattern', 'Not applicable based on skipping pattern'},
        'None (Bonded)': {'None (Bonded)', 'None (Bonded to Sponsoring Organisation)', '4'},
        'None (Self-employed)': {'None (Self-employed)', 'None (Self Employed)', '2'},
        'None (Family business)': {'None (Family business)', 'None (Family Business)', '3'},
        'Refused': {'Refused'},
        'None': {'None', '1.0', '1'}
    },
    'offer_wait': {
        'Less than 1 month': {'Less than 1 month', 'Less than one month', '1', '1.0'},
        '1 to less than 3 months': {'1 to less than 3 months', 'One month to less than 3 months',
                                    'One to less than three months', 'One month to less than three months', '2', '2.0'},
        '3 to less than 6 months': {'3 to less than 6 months', '3 months to less than 6 months',
                                    'Three to less than six months', 'Three months to less than six months', '3',
                                    '3.0'},
        '6 months or more': {'6 months or more', 'Six months and more', 'Six months or more', '4', '4.0'},
        'Nil': {'.', '#NULL!'}
    },
    'main_channel': {
        'Nil': {'.', 'Did not specify'},
        'Internet/Website': {
            'Internet/Website (e.g. job portals, social media networks et', '12.0', '6',
            'Internet/Website (e.g. job portals, social media networks etc.)', 'Q21_8 Other job websites\\ portals'
        },
        'NUS Career Centre': {
            'NUS Career Centre (e.g. NUS Career Fair, eCareer Fair, recru', '8', '3.0',
            'NUS Career Centre (e.g. NUS Career Fair, NUS Jobs Connect, R',
            'NUS Career Centre (e.g. NUS Career Fair, NUS Jobs Connect, Referral by staff from NUS Career Centre, '
            'recruitment talks,', 'Centre for Future-ready Graduates (Previously known as NUS C',
            'Q21_2 Centre for Future-ready Graduates Career Events'
        },
        'Traditional Channels': {
            'Traditional channels (e.g. newspapers advertisements, recomm', '2', '6.0',
            'Traditional channels (e.g. newspapers advertisements, recommendations by family members and friends, '
            'etc.)', 'Q21_11 Newspaper\\ magazine advertisements'
        },
        'Others': {'Others', '9', '13.0', 'Chanced upon the job', 'Q21_14 Others, please specify'},
        'Refused': {'Refused', 'Refused to comment', '10.0', 'No comments'},
        'Industrial Attachment': {
            'Faculty/School-level industrial attachment, professional att', '1.0', '3',
            'Faculty/School-level industrial attachment, professional attachment or professional internship',
            'Q21_6 Faculty\\ School-level industrial attachment, professional attachment, or professional internship'
        },
        'Career Services': {
            'Faculty/School-level career services (e.g. Business School C', '2.0', '4',
            'Faculty/School-level career services (e.g. Business School Career Services)',
            'Q21_5 Faculty\\ School-level career services'
        },
        'Referral': {
            'Referral or facilitation by an individual faculty or staff m', '5', '4.0',
            'Referral or facilitation by an individual faculty or staff member of NUS',
            'Q21_13 Referrals by family members and friends',
            'Q21_7 Referral or facilitation by an individual faculty or staff member of NUS (excluding CFG staff)'
        },
        'Employment Agencies': {
            'Employment agencies and services', '1', '5.0', 'Q21_10 Employment\\ recruitment agencies'
        },
        'NUS Career Centre (Online)': {
            'NUS Career Centre (eJob Centre)', '7', '11.0',
            'NUS Career Centre (NUS TalentConnect, previously known as eJ',
            'NUS Career Centre (NUS TalentConnect, previously known as eJob Centre)',
            'Q21_1 Centre for Future-ready Graduates Job Portal (NUS Talent Connect) & E News'
        },
        'Direct Application': {
            'Direct application to company', '9.0', 'Direct application to the company', 'Applied direct to companies'
        },
        'Scholarship Bond': {
            'Scholarship Bond / Award', '8.0', 'Scholarship bond / award', 'Scholarship bond', 'Scholarship bonds'
        },
        'Company Invitation': {
            'Invited to apply by company', '7.0', 'Invited by company to apply', 'Approached directly by company',
            'Invited to apply by company / headhunted'
        },
        'Academic Conference': {
            'Academic Conference', 'Conferences', 'Through competition / conferences organised by company'
        },
        'Conversion': {
            'Conversion to full time from internship / contract work / training',
            'Previous workplace / from internship', 'Conversion from internship / part time jobs / training'
        },
        'External Career Fair': {
            'Career Fair / Talks (Outside of University)', 'Career talks / fairs outside of university',
            'Q21_9 Other career fairs'
        },
        'Self-employed': {'Sourced from NS time / self-employed'},
        'Social Media': {'Q21_12 Social media networks'},
        'CFG Internships': {'Q21_3 Centre for Future-ready Graduates Internships'},
        'CFG Coaching': {'Q21_4 Centre for Future-ready Graduates Coaching and Advisory'}
    },
    'channel_nus_portal': _GENERAL_BINARY_EQUIV,
    'channel_cfg_posting': _GENERAL_BINARY_EQUIV,
    'channel_nus_events': _GENERAL_BINARY_EQUIV,
    'channel_ia': _GENERAL_BINARY_EQUIV,
    'channel_nus_ref': _GENERAL_BINARY_EQUIV,
    'channel_staff_ref': _GENERAL_BINARY_EQUIV,
    'channel_fr_ref': _GENERAL_BINARY_EQUIV,
    'channel_other_web': _GENERAL_BINARY_EQUIV,
    'channel_social_media': _GENERAL_BINARY_EQUIV,
    'channel_agency': _GENERAL_BINARY_EQUIV,
    'channel_traditional': _GENERAL_BINARY_EQUIV,
    'channel_ext_cf': _GENERAL_BINARY_EQUIV,
    'channel_no_need': _GENERAL_BINARY_EQUIV,
    'channel_other': _GENERAL_BINARY_EQUIV,
    'working_country': {
        'Singapore': {'Singapore', '1', '1.0'},
        'USA': {'USA', '2', '2.0'},
        'Canada': {'Canada', '3', '3.0'},
        'Malaysia': {'Malaysia', '5', '5.0'},
        'Vietnam': {'Vietnam', '4', '4.0'},
        'Japan': {'Japan', '6', '6.0'},
        'Mainland China': {'Mainland China', '7', '7.0'},
        'Hong Kong SAR': {'Hong Kong SAR', '8', '8.0'},
        'Others': {'Others <i>(Please specify)</i>', '14', '13', '12', '11', '10', '9', '9.0', 'Others',
                   'Others, please specify country'},
        'Nil': {'.', 'Not applicable based on skipping pattern', 'Not applicable'}
    } | {name: {name} for name in {'Indonesia', 'Australia', 'Taiwan', 'Germany', 'Thailand', 'Dubai',
                                   'Switzerland', 'United Kingdom', 'South Korea', 'India'}},
    'is_overseas': {
        'Y': {'Overseas', 'Overseas affiliates of Singapore registered or incorporated company/ organisation',
              'Others, please specify:'},
        'N': {'Singapore', 'Local', 'Non Applicable based on skip pattern'},
    },
    'overseas_type': {
        'Singapore registered': {
            'Overseas affiliates of Singapore registered or incorporated', '1', '1.0',
            'Overseas affiliates of Singapore registered or incorporated company/ organisation',
            'Overseas affiliates of Singapore registered or incorporated company \\ organisation',
            'Overseas affiliates of Singapore registered or incorporated company\\ organisation'
        },
        'Others': {'99', '2.0', '2', '3', '4', '5', '6', 'Others', 'Others, please specify:', 'Others, please specify'},
        'Nil': {'.', 'Non Applicable based on skip pattern', 'Not applicable based on skipping pattern'}
    },
    'return_sg': _GENERAL_BINARY_EQUIV,
    'helped_by_course': _GENERAL_INC_RATE_EQUIV,
    'helped_by_nus_brand': _GENERAL_INC_RATE_EQUIV,
    'course_related': {
        'Related': {'1', '1.0', 'Yes, I am employed in a job related to my course of study'},
        'Partial': {'2', '2.0', 'Yes, I am employed in a job partially related to my course o',
                    'Yes, I am employed in a job partially related to my course of study'},
        'No': {'3', '3.0', 'No, I am employed in a job not related to my course of study'},
        'Nil': {'.', 'Non Applicable based on skip pattern', 'Not applicable based on skipping pattern'}
    },
    'unrelated_reason': {
        'Not offered': {
            '1', '1.0', 'Applied for training-related job, but did not get an offer',
            'Applied for job related to course of study, but did not get an offer'
        },
        'Stronger interest': {'2', '2.0', 'Stronger interest in current job'},
        'Better prospects': {'3', '3.0', 'Better career prospects in current job'},
        'Better pay': {'4', '4.0', 'Better pay in current job', 'Financial reasons'},
        'Influenced by media': {
            '5', '5.0', 'Influenced by media', 'Media portrayal of job', 'Media portrayal of current job'
        },
        'Influenced by people': {
            '6', '6.0', 'Influenced by parents, relatives and/or friends', 'Advice from family and friends'
        },
        'Others': {'Others', '7', '7.0', 'Others, please specify'},
        'Temporary measure': {
            'Temporarily measure while looking for a full time job', 'Temporarily measure while deciding what to do',
            'Temporarily measure while waiting for full time job to commence', 'Temporary / Part time job',
            'Waiting for new job / further studies', 'Temporary measure while looking for FTP',
            'Preparing for futher studies'
        },
        'Try new': {'To try out something new', 'Try something new / explore new areas'},
        'Few openings': {
            'Few job openings in related areas', 'Limitied choices / opportunities in field of study',
            'Few job openings in related area'
        },
        'First offer': {
            'Accepted the first job offer that came along', 'Accepted the first job that came along',
            'Offered a job so just took it / Limited jobs in the job mark', 'Current job is good / convenient'
        },
        'Prior experience': {
            'Worked in this area previously', 'Already got the job before studying', 'Worked on this area previously',
            'Did internship at the company previously'
        },
        'Related to other degree': {'Related to Diploma / Second degree',  'Related to diploma / second degree'},
        'Placed position': {
            'Placed on the job which was different from what was applied for',
            'Thought was related but turned out otherwise'
        },
        'Serving bond': {
            'Serving scholarship bond', 'Scholarship / Serving bond', 'As part of programme',
            'Serving scholarshop bond', 'Bonded/ Scholarship'
        },
        'Gain experience': {'To gain some work experience first', 'Gain more experience / skills'},
        'Family/own business': {
            'Helping out in family business', 'Family business / Self-employed', 'Working in own company'
        },
        'Refused': {'Refused to disclose'},
        'Personal choice': {'Personal choice', 'No reasons', 'Personal Choice'},
        'Course too general': {'Course is too general'},
        'Nil': {'Non Applicable based on skip pattern', 'Not applicable based on skipping pattern', '.'}
    },
    'unrelated_lack': _GENERAL_BINARY_EQUIV,
    'unrelated_failed': _GENERAL_BINARY_EQUIV,
    'unrelated_interest': _GENERAL_BINARY_EQUIV,
    'unrelated_strength': _GENERAL_BINARY_EQUIV,
    'unrelated_pay': _GENERAL_BINARY_EQUIV,
    'unrelated_opportunity': _GENERAL_BINARY_EQUIV,
    'unrelated_prospects': _GENERAL_BINARY_EQUIV,
    'unrelated_co_worker': _GENERAL_BINARY_EQUIV,
    'unrelated_balance': _GENERAL_BINARY_EQUIV,
    'unrelated_convenience': _GENERAL_BINARY_EQUIV,
    'unrelated_social_status': _GENERAL_BINARY_EQUIV,
    'unrelated_support': _GENERAL_BINARY_EQUIV,
    'unrelated_environment': _GENERAL_BINARY_EQUIV,
    'unrelated_others': _GENERAL_BINARY_EQUIV,
    'attend_career': _GENERAL_BINARY_EQUIV,
    'attend_faculty': _GENERAL_BINARY_EQUIV,
    'prepare_written_comm': _GENERAL_INC_RATE_EQUIV,
    'prepare_oral_comm': _GENERAL_INC_RATE_EQUIV,
    'prepare_multidisciplinary': _GENERAL_INC_RATE_EQUIV,
    'prepare_international': _GENERAL_INC_RATE_EQUIV,
    'prepare_org': _GENERAL_INC_RATE_EQUIV,
    'prepare_critical': _GENERAL_INC_RATE_EQUIV,
    'prepare_creative': _GENERAL_INC_RATE_EQUIV,
    'prepare_learn_ind': _GENERAL_INC_RATE_EQUIV,
    'prepare_interpersonal': _GENERAL_INC_RATE_EQUIV,
    'prepare_personal': _GENERAL_INC_RATE_EQUIV,
    'prepare_cross_cultural': _GENERAL_INC_RATE_EQUIV,
    'prepare_change_env': _GENERAL_INC_RATE_EQUIV,
    'prepare_career': _GENERAL_INC_RATE_EQUIV,
    'prepare_domain': _GENERAL_INC_RATE_EQUIV
}
_GES_COLUMN_DTYPES: Dict[str, Union[Type, str]] = {
    'student_token': str,
    'academic_load': 'string',
    'gender': 'string',
    'degree': 'string',
    'major': 'string',
    'usp': 'string',
    'faculty': 'string',
    'noc': 'string',
    'segment_als': 'string',
    'hon': 'string',
    'class_of_hons': 'string',
    'confer_date': 'datetime64[ns]',
    'progtype': 'string',
    'first_choice': 'string',
    'is_first_choice': 'string',
    'professional_degree': 'string',
    'residency': 'string',
    'fin_src': 'string',
    'race': 'string',
    'new_citizen': 'string',
    'primary_school_in_sg': 'string',
    'secondary_school_in_sg': 'string',
    'ite_in_sg': 'string',
    'jc_mi_in_sg': 'string',
    'poly_in_sg': 'string',
    'none_edu_in_sg': 'string',
    'final_exam': 'datetime64[ns]',
    'activity_status': 'string',
    'not_ft_perm_reason': 'string',
    'employment_status': 'string',
    'company_type': 'string',
    'industry': 'string',
    'basic_salary': 'float32',
    'gross_salary': 'float32',
    'ot_salary': 'float32',
    'start_search': 'datetime64[ns]',
    'gap_grad_search': 'string',
    'offer_date': 'datetime64[ns]',
    'offer_count': 'string',
    'offer_wait': 'string',
    'main_channel': 'string',
    'channel_nus_portal': 'string',
    'channel_cfg_posting': 'string',
    'channel_nus_events': 'string',
    'channel_ia': 'string',
    'channel_nus_ref': 'string',
    'channel_staff_ref': 'string',
    'channel_fr_ref': 'string',
    'channel_other_web': 'string',
    'channel_social_media': 'string',
    'channel_agency': 'string',
    'channel_traditional': 'string',
    'channel_ext_cf': 'string',
    'channel_no_need': 'string',
    'channel_other': 'string',
    'working_country': 'string',
    'is_overseas': 'string',
    'overseas_type': 'string',
    'return_sg': 'string',
    'helped_by_course': 'string',
    'helped_by_nus_brand': 'string',
    'course_related': 'string',
    'unrelated_reason': 'string',
    'unrelated_lack': 'string',
    'unrelated_failed': 'string',
    'unrelated_interest': 'string',
    'unrelated_strength': 'string',
    'unrelated_pay': 'string',
    'unrelated_opportunity': 'string',
    'unrelated_prospects': 'string',
    'unrelated_co_worker': 'string',
    'unrelated_balance': 'string',
    'unrelated_convenience': 'string',
    'unrelated_social_status': 'string',
    'unrelated_support': 'string',
    'unrelated_environment': 'string',
    'unrelated_others': 'string',
    'attend_career': 'string',
    'attend_faculty': 'string',
    'prepare_written_comm': 'string',
    'prepare_oral_comm': 'string',
    'prepare_multidisciplinary': 'string',
    'prepare_international': 'string',
    'prepare_org': 'string',
    'prepare_critical': 'string',
    'prepare_creative': 'string',
    'prepare_learn_ind': 'string',
    'prepare_interpersonal': 'string',
    'prepare_personal': 'string',
    'prepare_cross_cultural': 'string',
    'prepare_change_env': 'string',
    'prepare_career': 'string',
    'prepare_domain': 'string'
}

_degree2major_map: Dict[str, str] = {
    'Bachelor of Engineering (Electrical Engineering)': 'Electrical Engineering',
    'Bachelor of Business Administration': 'BBA',
    'Bachelor of Engineering (Mechanical Engineering)': 'Mechanical Engineering',
    'Bachelor of Engineering (Chemical Engineering)': 'Chemical Engineering',
    'Bachelor of Laws': 'Law',
    'Bachelor of Science (Real Estate)': 'Real Estate',
    'Bachelor of Medicine and Bachelor of Surgery': 'Medicine',
    'Bachelor of Science (Project and Facilities Management)': 'Project & Facilities Management',
    'Bachelor of Science (Pharmacy)': 'Pharmacy',
    'Bachelor of Technology (Electronics Engineering)': 'Electronics Engineering',
    'Bachelor of Computing (Information Systems)': 'Information Systems',
    'Bachelor of Engineering (Computer Engineering)': 'Computer Engineering',
    'Bachelor of Business Administration (Accountancy)': 'Accountancy',
    'Bachelor of Engineering (Civil Engineering)': 'Civil Engineering',
    'Bachelor of Engineering (Industrial and Systems Engineering)': 'Industrial & System Engineering',
    'Bachelor of Engineering (Biomedical Engineering)': 'Biomedical Engineering',
    'Bachelor of Computing (Electronic Commerce)': 'E-Commerce',
    'Bachelor of Technology (Mechanical Engineering)': 'Mechanical Engineering',
    'Bachelor of Computing (Computer Science)': 'Computer Science',
    'Bachelor of Computing (Communications and Media)': 'Communications & Media',
    'Bachelor of Science (Nursing)': 'Nursing',
    'Bachelor of Dental Surgery': 'Dental Surgery',
    'Bachelor of Engineering (Materials Science and Engineering)': 'Materials Science & Engineering',
    'Bachelor of Engineering (Environmental Engineering)': 'Environmental Engineering',
    'Bachelor of Engineering (Engineering Science)': 'Engineering Science',
    'Bachelor of Arts (Industrial Design)': 'Industrial Design',
    'Bachelor of Arts (Architecture)': 'Architecture',
    'Bachelor of Science (Computational Biology)': 'Computational Biology',
    'Bachelor of Technology (Chemical Engineering)': 'Chemical Engineering',
    'Bachelor of Technology (Manufacturing Engineering)': 1,
    'Bachelor of Technology (Industrial and Management Engineering)': 'Industrial & Management Engineering',
    'Bachelor of Science (Business Analytics)': 1,
    'Bachelor of Computing (Information Security)': 1,
    'Bachelor of Science (Data Science and Analytics)': 1
}
_program2faculty_map: Dict[str, str] = {
    'Bachelor of Technology': 'School of Cont & Lifelong Edun',
    'Bachelor of Applied Science': 'Faculty of Science',
    'Bachelor of Arts': 'Faculty of Arts & Social Sciences',
    'Bachelor of Business Administration': 'NUS Business School',
    'Bachelor of Computing': 'School of Computing',
    'Bachelor of Dental Surgery': 'Faculty of Dentistry',
    'Bachelor of Engineering': 'Faculty of Engineering',
    'Bachelor of Environmental Studies': 'School of Design & Environment',
    'Bachelor of Medicine and Bachelor of Surgery': 'YLL School of Medicine',
    'Bachelor of Laws': 'Faculty of Law',
    'Bachelor of Music': 'YST Conservatory of Music',
    'Bachelor of Science': 'Faculty of Science'
}


def _get_canonical_value(col: str, original: Any) -> Any:
    if col not in _GES_COLUMN_VALUE_EQUIV:
        return original
    if pd.isnull(original):
        return original
    for equiv_class, values in _GES_COLUMN_VALUE_EQUIV[col].items():
        if original in values:
            return equiv_class
    return original


def _combine_columns(col1: str, col2: str, df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda row: row[col1] if not pd.isnull(row[col1]) else row[col2], axis=1
    )


def _program2faculty(program: str) -> str:
    if '(' in program:
        program = program[:program.index('(')-1]
    if program in _program2faculty_map:
        return _program2faculty_map[program]
    return ''


def _process_year(year: int, df: pd.DataFrame) -> pd.DataFrame:
    result = df.rename(columns={
        col: col_by_year[year]
        for col, col_by_year in _GES_COLUMNS.items()
        if year in col_by_year
    }).astype(_GES_COLUMN_DTYPES)
    result = result[[col for col in _GES_COLUMN_DTYPES if col in set(result.columns)]]
    all_columns = set(result.columns)

    result.loc[:, 'usp'] = result['usp'].fillna('N')
    for col in result.columns:
        if _GES_COLUMN_DTYPES[col] == 'string':
            result.loc[:, col] = result[col].apply(lambda x: x if pd.isnull(x) else
                                                   np.nan if x.strip() == '' else x.strip())
        result.loc[:, col] = result[col].apply(partial(_get_canonical_value, col=col))

    result.loc[:, 'major'] = result.apply(
        lambda row: row['major'] if not pd.isnull(row['major']) and row['major'] != 'Nil'
        else _degree2major_map[row['degree']] if row['degree'] in _degree2major_map else np.nan, axis=1
    )
    if 'first_choice' in all_columns:
        result.loc[:, 'first_choice'] = result['first_choice'].apply(partial(_get_canonical_value, col='major'))
        result.loc[:, 'first_choice'] = result.apply(
            lambda row: row['first_choice'] if not pd.isnull(row['first_choice'])
            else row['major'] if row['is_first_choice'] == 'Y' else np.nan, axis=1
        )
    if 'is_first_choice' not in all_columns and 'first_choice' in all_columns:
        result.loc[:, 'is_first_choice'] = result.apply(
            lambda row: 'Y' if row['first_choice'] == row['major'] else 'N', axis=1
        )
    for col in ['not_ft_perm_reason', 'employment_status', 'company_type', 'industry', 'offer_count', 'main_channel',
                'channel_nus_portal', 'channel_cfg_posting', 'channel_nus_events', 'channel_ia', 'channel_nus_ref',
                'channel_staff_ref', 'channel_fr_ref', 'channel_other_web', 'channel_social_media', 'channel_agency',
                'channel_traditional', 'channel_ext_cf', 'channel_no_need', 'channel_other', 'working_country',
                'is_overseas', 'overseas_type', 'return_sg', 'helped_by_course', 'helped_by_nus_brand',
                'course_related', 'unrelated_reason', 'unrelated_lack', 'unrelated_failed', 'unrelated_interest',
                'unrelated_strength', 'unrelated_pay', 'unrelated_opportunity', 'unrelated_prospects',
                'unrelated_co_worker', 'unrelated_balance', 'unrelated_convenience', 'unrelated_social_status',
                'unrelated_support', 'unrelated_environment', 'unrelated_others', 'attend_career', 'attend_faculty',
                'prepare_written_comm', 'prepare_oral_comm', 'prepare_multidisciplinary', 'prepare_international',
                'prepare_org', 'prepare_critical', 'prepare_creative', 'prepare_learn_ind', 'prepare_interpersonal',
                'prepare_personal', 'prepare_cross_cultural', 'prepare_change_env', 'prepare_career', 'prepare_domain']:
        result.loc[:, col] = result[col].replace({'Nil': np.nan})
    result = result.astype({
        col: 'Int32' for col in [
            'helped_by_course', 'helped_by_nus_brand', 'prepare_written_comm', 'prepare_oral_comm',
            'prepare_multidisciplinary', 'prepare_international', 'prepare_org', 'prepare_critical', 'prepare_creative',
            'prepare_learn_ind', 'prepare_interpersonal', 'prepare_personal', 'prepare_cross_cultural',
            'prepare_change_env', 'prepare_career', 'prepare_domain'
        ]
    })
    if 'ot_salary' in all_columns and 'gross_salary' not in all_columns:
        result.loc[:, 'gross_salary'] = result['basic_salary'] + result['ot_salary']
    if 'ot_salary' not in all_columns and 'gross_salary' in all_columns:
        result.loc[:, 'ot_salary'] = result['gross_salary'] - result['basic_salary']
    if 'working_country' in all_columns and 'is_overseas' not in all_columns:
        result.loc[:, 'is_overseas'] = result['working_country'].apply(
            lambda x: np.nan if pd.isnull(x) else 'N' if x == 'Singapore' else 'Y'
        )
    return result


def ges2010(src: pd.DataFrame) -> pd.DataFrame:
    result = src.copy()
    result.loc[:, 'hon'] = result['degree'].apply(
        lambda x: 'Y' if 'Hon' in x or x.startswith('B.Eng.') or x.startswith('B.Comp.')
                  or x.strip() in {'B.D.S.', 'M.B.B.S.'} or x.startswith('B.A. (') else 'N'
    )
    result.loc[:, 'ans_02'] = result.apply(
        lambda row: 14 if row['ans_02a'] == 1 else np.nan if pd.isnull(row['ans_02a']) and pd.isnull(row['ans_02b'])
        else 13 if row['ans_02b'] == 12 else row['ans_02b'], axis=1
    )
    for i in range(1, 7):
        result.loc[:, f'ans_03_{i}'] = result[f'ans_03_{i}'].replace({2: 'N', 1: 'Y'})
    result.loc[:, 'final_exam'] = result.apply(
        lambda row: np.nan if pd.isnull(row['ans_04_mth']) else
        f'{int(row["ans_04_yr"])}-{int(row["ans_04_mth"])}-01', axis=1
    )
    result.loc[:, 'activity_status'] = result.apply(
        lambda row: np.nan if pd.isnull(row['ans_06']) else {
            1: 'Working Full-Time in a Permanent job',
            2: 'Working Full-Time in a Temporary job',
            3: 'Working Part-Time in a Permanent job',
            4: 'Working Part-Time in a Temporary job'
        }[row['ans_07']] if row['ans_06'] == 1 else row['ans_06'], axis=1
    )
    result.loc[:, 'offer_count'] = _combine_columns('ans_19', 'ans_37', result)
    return _process_year(2010, result)


def ges2011(src: pd.DataFrame) -> pd.DataFrame:
    result = src.copy()
    result.loc[:, 'academic_load'] = result.apply(
        lambda row: 'Full-Time' if str(row['type']).strip() == '1' and str(row['gradtype']).strip() == '1'
        else 'Part-Time' if str(row['type']).strip() == '2' else 'Follow-Up', axis=1
    )
    result.loc[:, 'onnocduringcandidature_org'] = result['onnocduringcandidature_org'].fillna('N')
    result.loc[:, 'q2'] = result['q2'].replace({'.': np.nan})
    result.loc[:, 'activity_status'] = result.apply(
        lambda row: np.nan if pd.isnull(row['q4']) else {
            1: 'Working Full-Time in a Permanent job',
            2: 'Working Full-Time in a Temporary job',
            3: 'Working Part-Time in a Permanent job',
            4: 'Working Part-Time in a Temporary job'
        }[row['q5']] if row['q4'] == 1 else row['q4'], axis=1
    )
    result.loc[:, 'q16'] = result['q16'].apply(
        lambda x: x if x.strip() == '.' else str(x).zfill(5)
    )
    result.loc[:, 'q12a'] = result['q12a'].replace({'.': np.nan})
    result.loc[:, 'q12b'] = result['q12b'].replace({'.': np.nan})
    result.loc[:, 'offer_count'] = _combine_columns('q14', 'q27', result)
    result.loc[:, 'q17a'] = result['q17a'].replace({'.': np.nan})
    result.loc[:, 'q17b'] = result['q17b'].replace({'.': np.nan})
    return _process_year(2011, result)


def ges2012(src: pd.DataFrame) -> pd.DataFrame:
    result = src.copy()
    result.loc[:, 'hon'] = result['degree'].apply(
        lambda x: 'Y' if any([v in x.lower() for v in {
            'hon', 'engi', 'med', 'law', 'real', 'phar', 'arch', 'info', 'comp', 'sur',
            'industrial design', 'comm'
        }]) else 'N'
    )
    result.loc[:, 'confer_date'] = result.apply(
        lambda row: np.nan if pd.isnull(row['cd_year']) else
        f'{row["cd_mth"]} {int(row["cd_year"])}', axis=1
    )
    result.loc[:, 'start_search'] = result.apply(
        lambda row: datetime.strptime(row['r_q7'], '%b-%y').strftime('%Y-%m-%d') if isinstance(row['r_q7'], str) else
        np.nan if pd.isnull(row['q24_1']) else f'{int(row["q24_2"])}-{row["q24_1"]}-01', axis=1
    )
    result.loc[:, 'offer_date'] = result.apply(
        lambda row: datetime.strptime(row['r_q8'], '%b-%y').strftime('%Y-%m-%d') if isinstance(row['r_q8'], str) else
        np.nan if pd.isnull(row['q25_1']) else f'{int(row["q25_2"])}-{row["q25_1"]}-01', axis=1
    )
    result.loc[:, 'offer_count'] = _combine_columns('q17', 'q23', result)
    result.loc[:, 'offer_wait'] = result.apply(
        lambda row: row['r_mth_q8_cd'] if not pd.isnull(row['r_mth_q8_cd'])
        else np.nan if pd.isnull(row['mth_q25_q24']) else
        '6 months or more' if row['mth_q25_q24'] >= 6 else
        '3 to less than 6 months' if 3 <= row['mth_q25_q24'] < 6 else
        '1 to less than 3 months' if 1 <= row['mth_q25_q24'] < 3 else
        'Less than 1 month', axis=1
    )
    return _process_year(2012, result)


def ges2013(src: pd.DataFrame) -> pd.DataFrame:
    result = src.copy()
    result.loc[:, 'offer_wait'] = result['q19_q20_difference'].apply(
        lambda x: '6 months or more' if x >= 6 else
        '3 to less than 6 months' if 3 <= x < 6 else
        '1 to less than 3 months' if 1 <= x < 3 else
        'Less than 1 month'
    )
    result.loc[:, 'q21'] = result.apply(
        lambda row: row['q21'] if row['q21'] != 'Others'
        else row['q21_other'], axis=1
    )
    return _process_year(2013, result)


def ges2014(src: pd.DataFrame) -> pd.DataFrame:
    result = src.copy()
    result.loc[:, 'start_search'] = result.apply(
        lambda row: f'{int(row["q20_year"])}-{int(row["q20_month"])}-01', axis=1
    )
    result.loc[:, 'offer_date'] = result.apply(
        lambda row: f'{int(row["q19_year"])}-{int(row["q19_month"])}-01', axis=1
    )
    return _process_year(2014, result)


def ges2015(src: pd.DataFrame) -> pd.DataFrame:
    result = src.copy()
    result.loc[:, 'degree'] = _combine_columns('nus_programme', 'combineprogramme', src)
    result.loc[:, 'major'] = src.apply(
        lambda row: row['major1_1'] if not pd.isnull(row['major1_1'])
        and row['major1_1'].strip() != '' else row['major'], axis=1
    )
    result.loc[:, 'faculty'] = src.apply(
        lambda row: row['nus_faculty'] if not pd.isnull(row['nus_faculty'])
        else _program2faculty(row['combineprogramme']), axis=1
    )
    result.loc[:, 'nus_honoursplan'] = src['nus_honoursplan'].replace({'1': 'A1', '3': 'A3'})
    result.loc[:, 'd2'] = src.apply(
        lambda row: row['d2'] if not pd.isnull(row['d2']) and row['d2'].strip() != ''
        else row['nationality'], axis=1
    )
    result.loc[:, 'q10_basic'] = result['q10_basic'].apply(lambda x: x.replace(',', '') if isinstance(x, str) else x)
    result.loc[:, 'q10_gross'] = result['q10_gross'].apply(lambda x: x.replace(',', '') if isinstance(x, str) else x)
    result.loc[:, 'q10_ot'] = result['q10_ot'].apply(lambda x: x.replace(',', '') if isinstance(x, str) else x)
    return _process_year(2015, result)


def ges2016(src: pd.DataFrame) -> pd.DataFrame:
    result = src.copy()
    result.loc[:, 'hon'] = result['honours_class_1st_major'].apply(
        lambda x: 'Y' if 'Hon' in x else 'N'
    )
    result.loc[:, 'start_search'] = result.apply(
        lambda row: f'{"MAY" if pd.isnull(row["start_of_job_search_month"]) else row["start_of_job_search_month"]} '
                    f'{2016 if pd.isnull(row["start_of_job_search_year"]) else int(row["start_of_job_search_year"])}'
        if not pd.isnull(row['start_of_job_search_month']) and not pd.isnull(row['start_of_job_search_year'])
        else np.nan, axis=1
    )
    term2mth = {
        0: 'May',
        10: 'Dec',
        20: 'May',
        30: 'Jul',
        40: 'Aug'
    }
    result.loc[:, 'grad_date'] = result['expected_graduation_term'].apply(
        lambda x: np.nan if pd.isnull(x) else datetime.strptime(f'{x // 100}-{term2mth[x % 100]}', '%y-%b')
        .strftime('%Y-%m-%d')
    ).astype('datetime64[ns]')
    result.loc[:, 'gap_grad_search'] = result.apply(
        lambda row: np.nan if pd.isnull(row['start_search'])
        else str(max(-30, min(30, (row['start_search'] - row['grad_date']).days // 7))), axis=1
    )
    result.loc[:, 'offer_date'] = result.apply(
        lambda row: f'{"January" if pd.isnull(row["date_of_1st_offer_month"]) else row["date_of_1st_offer_month"]} '
                    f'{2016 if pd.isnull(row["date_of_1st_offer_year"]) else int(row["date_of_1st_offer_year"])}'
        if not pd.isnull(row['date_of_1st_offer_month']) and not pd.isnull(row['date_of_1st_offer_year'])
        else np.nan, axis=1
    )
    result.loc[:, 'channel_services'] = result.apply(
        lambda row: 'Yes' if row['cfg_career_advisory'] == 'Yes' or row['faculty_career_services'] == 'Yes'
        else 'No' if not pd.isnull(row['cfg_career_advisory']) and not pd.isnull(row['faculty_career_services'])
        and not (len(row['cfg_career_advisory']) > 5 or len(row['faculty_career_services']) > 5)
        else 'Non Applicable based on skip pattern', axis=1
    )
    result.loc[:, 'channel_ia'] = result.apply(
        lambda row: 'Yes' if row['cfg_internships'] == 'Yes' or row['faculty_internships'] == 'Yes'
        else 'No' if not pd.isnull(row['cfg_internships']) and not pd.isnull(row['faculty_internships'])
        and not (len(row['cfg_internships']) > 5 or len(row['faculty_internships']) > 5)
        else 'Non Applicable based on skip pattern', axis=1
    )
    return _process_year(2016, result)


def ges2017(src: pd.DataFrame) -> pd.DataFrame:
    result = src.copy()
    result.loc[:, 'q5'] = result.apply(
        lambda row: row['q5_emp'] if not row['q5_emp'].startswith('Not applicable')
        else row['q5_self'], axis=1
    )
    result.loc[:, 'q10_basic'] = result['q10_basic'].replace({'#NULL!': np.nan})
    result.loc[:, 'q10_overtime'] = result['q10_overtime'].replace({'#NULL!': np.nan})
    result.loc[:, 'q20_date'] = result['q20_date'].replace({'#NULL!': np.nan})
    result.loc[:, 'q19_date'] = result['q19_date'].replace({'#NULL!': np.nan})
    result.loc[:, 'q21_other'] = result.apply(
        lambda row: 'Yes' if any(row[f'q21_{i}'] == 'Yes' for i in range(13, 22))
        else 'No' if all(not pd.isnull(row[f'q21_{i}']) and len(row[f'q21_{i}']) <= 5 for i in range(13, 22))
        else 'Not', axis=1
    )
    return _process_year(2017, result)


def ges2018(src: pd.DataFrame) -> pd.DataFrame:
    result = src.copy()
    result.loc[:, 'noc'] = result['noc'].fillna('N')
    result.loc[:, 'hon'] = result['honours_descr'].apply(
        lambda x: 'Y' if 'Hon' in x else 'N'
    )
    result.loc[:, 'q5'] = result.apply(
        lambda row: row['q5_employee'] if not row['q5_employee'].startswith('Not applicable')
        else row['q5_self'], axis=1
    )
    return _process_year(2018, result)


def ges2019(src: pd.DataFrame) -> pd.DataFrame:
    result = src.copy()
    result.loc[:, 'degree'] = _combine_columns('conferreddegreedescr', 'conferreddegreedescr1', src)
    result.loc[:, 'usp'] = _combine_columns('uspscholaryorn', 'usp', src)
    result.loc[:, 'faculty'] = _combine_columns('faculty', 'facultydescr', src)
    result.loc[:, 'hon'] = _combine_columns('honours_descr', 'honours_descr1', src).apply(
        lambda x: np.nan if pd.isnull(x) or x.strip() == '' else 'Y' if 'Hon' in x else 'N'
    )
    result.loc[:, 'class_of_hons'] = _combine_columns('honours_descr', 'honours_descr1', src)
    result.loc[:, 'residency'] = _combine_columns('residency', 'residencydescr', src)
    result.loc[:, 'start_search'] = result.apply(
        lambda row: np.nan if pd.isnull(row['q22_month']) and pd.isnull(row['q22_year']) else
        f'{int(row["q22_year"]) if not pd.isnull(row["q22_year"]) else 2019}-'
        f'{row["q22_month"] if not pd.isnull(row["q22_month"]) else "May"}-01',
        axis=1
    )
    result.loc[:, 'offer_date'] = result.apply(
        lambda row: np.nan if pd.isnull(row['q25_month']) and pd.isnull(row['q25_year']) else
        f'{int(row["q25_year"]) if not pd.isnull(row["q25_year"]) else 2019}-'
        f'{row["q25_month"] if not pd.isnull(row["q25_month"]) else "January"}-01',
        axis=1
    )
    result.loc[:, 'q23_14'] = result.apply(
        lambda row: 'Yes' if row['q23_14'] == 'Yes' or row['q23_15'] == 'Yes'
        else 'No' if not pd.isnull(row['q23_14']) and not pd.isnull(row['q23_15'])
        and not (len(row['q23_14']) > 5 and len(row['q23_15']) > 5) else 'Not', axis=1
    )
    result.loc[:, 'q17_14'] = result.apply(
        lambda row: 'Yes' if any(row[f'q17_{i}'] == 'Yes' for i in range(14, 17))
        else 'No' if all(not pd.isnull(row[f'q17_{i}']) and len(row[f'q17_{i}']) <= 5 for i in range(14, 17))
        else 'Not', axis=1
    )
    return _process_year(2019, result)


def ges2020(src: pd.DataFrame) -> pd.DataFrame:
    result = src.copy()
    result.loc[:, 'hon'] = src['honoursplandescr'].apply(
        lambda x: 'Y' if x.strip() != '' else 'N'
    )
    result.loc[:, 'q10_basic'] = result['q10_basic'].replace({' ': np.nan})
    result.loc[:, 'q10_overtime'] = result['q10_overtime'].replace({' ': np.nan})
    result.loc[:, 'start_search'] = result.apply(
        lambda row: row['q20_date'] if not pd.isnull(row['q20_date']) and str(row['q20_date']).strip() != ''
        else np.nan if (pd.isnull(row['q20_month']) or str(row['q20_month']).strip() == '') and
        (pd.isnull(row['q20_year']) or row['q20_year'].strip() == '') else
        f'{int(row["q20_year"]) if not pd.isnull(row["q20_year"]) and str(row["q20_year"]).strip() != "" else 2020} '
        f'{row["q20_month"] if not pd.isnull(row["q20_month"]) and str(row["q20_month"]).strip() != "" else "May"}',
        axis=1
    )
    result.loc[:, 'offer_date'] = result.apply(
        lambda row: row['q18_date'] if not pd.isnull(row['q18_date']) and str(row['q18_date']).strip() != ''
        else np.nan if (pd.isnull(row['q18_month']) or str(row['q18_month']).strip() == '') and
        (pd.isnull(row['q18_year']) or row['q18_year'].strip() == '') else
        f'{int(row["q18_year"]) if not pd.isnull(row["q18_year"]) and str(row["q18_year"]).strip() != "" else 2020} '
        f'{row["q18_month"] if not pd.isnull(row["q18_month"]) and str(row["q18_month"]).strip() != "" else "May"}',
        axis=1
    )
    return _process_year(2020, result)
