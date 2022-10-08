"""Student Information System tables."""

from typing import Dict, Any

import pandas as pd

from irg.schema import Table


def personal_data(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['student_token']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'primary_keys': ['student_token']
    }


def sis_academic_career(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['student_token']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        # 'determinants': [
        #     ['form_of_study', 'form_of_study_descr']
        # ],
        'primary_keys': ['student_token', 'academic_career'],
        'foreign_keys': [{
            'columns': ['student_token'],
            'parent': 'personal_data'
        }]
    }


def sis_academic_program_offer(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = []  # ['academic_program']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'ttype': 'base',
        'determinants': [
            ['academic_program', 'academic_program_descr']
        ],
        'primary_keys': ['academic_program']
    }


def sis_academic_program(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['student_token']  # ['student_token', 'academic_program', 'dual_academic_program']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'primary_keys': ['student_token', 'academic_career', 'academic_program'],
        'foreign_keys': [{
            'columns': ['student_token'],
            'parent': 'sis_academic_career'
        }, {
            'columns': ['academic_program'],
            'parent': 'sis_academic_program_offer',
        }, {
            'columns': ['dual_academic_program'],
            'parent': 'sis_academic_program_offer',
            'parent_columns': ['academic_program']
        }]
    }


def sis_plan_offer(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = []  # ['academic_plan']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'ttype': 'base',
        # 'determinants': [
        #     ['academic_plan', 'academic_plan_descr'],
        #     ['academic_plan_type', 'academic_plan_type_descr']
        # ],
        'primary_keys': ['academic_plan'],
    }


def sis_academic_plan(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['student_token']  # ['student_token', 'academic_program', 'academic_plan']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        # 'determinants': [
        #     ['degree', 'degree_descr'],
        # ],
        'primary_keys': ['student_token', 'academic_career', 'academic_program', 'academic_plan'],
        'foreign_keys': [{
            'columns': ['student_token', 'academic_career', 'academic_program'],
            'parent': 'sis_academic_program'
        }, {
            'columns': ['academic_plan'],
            'parent': 'sis_plan_offer',
        }]
    }


def sis_enrolment(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['student_token']  # ['student_token', 'academic_program', 'academic_plan']
    attributes = Table.learn_meta(src, id_cols, ['career_nbr'])
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        # 'determinants': [
        #     ['department', 'department_descr'],
        #     ['program_status', 'program_status_descr']
        # ],
        'primary_keys': ['student_token', 'academic_program', 'academic_plan', 'admit_tyear', 'admit_tsem'],
        'foreign_keys': [{
            'columns': ['student_token', 'academic_career', 'academic_program', 'academic_plan'],
            'parent': 'sis_academic_plan'
        }]
    }


def gym(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['student_token']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        # 'determinants': [
        #     ['degree', 'degree_descr'],
        # ],
        'primary_keys': ['student_token'],
        'foreign_keys': [{
            'columns': ['student_token'],
            'parent': 'personal_data'
        }]
    }


def sis_module_enrolment(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['student_token']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        # 'determinants': [
        #     ['degree', 'degree_descr'],
        # ],
        'primary_keys': ['student_token','module_code'],
        'foreign_keys': [{
            'columns': ['student_token'],
            'parent': 'personal_data'
        }]
    }

def sis_course(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = []
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        # 'determinants': [
        #     ['degree', 'degree_descr'],
        # ],
        'primary_keys': ['module_code'],
        'foreign_keys': [{
            'columns': ['module_code'],
            'parent': 'sis_module_enrolment'
        }]
    }

def sis_credits(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['student_token']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        # 'determinants': [
        #     ['degree', 'degree_descr'],
        # ],
        'primary_keys': ['student_token','module_code'],
        'foreign_keys': [{
            'columns': ['student_token'],
            'parent': 'sis_module_enrolment'
        }]
    }

def sis_milestone(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['student_token']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        # 'determinants': [
        #     ['degree', 'degree_descr'],
        # ],
        'primary_keys': ['student_token'],
        'foreign_keys': [{
            'columns': ['student_token'],
            'parent': 'personal_data'
        }]
    }

def academic_program_offer(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['academic_program']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'ttype': 'base',
        'determinants': [
            ['academic_program', 'academic_program_descr']
        ],
        'primary_keys': ['academic_program', 'tyear', 'tsem']
    }


def academic_plan_offer(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['academic_plan', 'academic_program']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'ttype': 'base',
        'determinants': [
            ['academic_plan', 'academic_plan_descr'],
            ['academic_plan_type', 'academic_plan_type_descr'],
        ],
        'primary_keys': ['academic_plan', 'academic_program', 'tyear', 'tsem'],
        'foreign_keys': [{
            'columns': ['academic_program', 'tyear', 'tsem'],
            'parent': 'academic_program_offer'
        }]
    }


def academic_subplan_offer(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['academic_plan', 'academic_program', 'academic_subplan']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'ttype': 'base',
        'determinants': [
            ['academic_subplan', 'academic_subplan_descr'],
        ],
        'primary_keys': ['academic_plan', 'academic_program', 'tyear', 'tsem', 'academic_subplan'],
        'foreign_keys': [{
            'columns': ['academic_plan', 'academic_program', 'tyear', 'tsem'],
            'parent': 'academic_plan_offer'
        }]
    }


def degree_offer(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['academic_plan', 'academic_program', 'department']
    num_cat_cols = ['faculty_code']
    attributes = Table.learn_meta(src, id_cols, num_cat_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'ttype': 'base',
        'determinants': [
            ['faculty_code', 'faculty_descr'],
            ['department', 'department_descr']
        ],
        'primary_keys': ['academic_plan', 'academic_program', 'tyear', 'tsem', 'degree', 'department'],
        'foreign_keys': [{
            'columns': ['academic_plan', 'academic_program', 'tyear', 'tsem'],
            'parent': 'academic_plan_offer'
        }]
    }


def student_enrolment(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = [
        'academic_plan', 'academic_program', 'department', 'student_token',
        'attached_to_ri', 'dual_academic_program', 'partner_university'
    ]
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'determinants': [
            ['attached_to_ri', 'attached_to_ri_descr'],
            ['form_of_study', 'form_of_study_descr'],
            ['partner_university', 'partner_university_descr']
        ],
        'primary_keys': ['academic_plan', 'academic_program', 'tyear', 'tsem', 'degree', 'department', 'student_token'],
        'foreign_keys': [{
            'columns': ['academic_plan', 'academic_program', 'tyear', 'tsem', 'degree', 'department'],
            'parent': 'degree_offer'
        }, {
            'columns': ['dual_academic_program', 'tyear', 'tsem'],
            'parent': 'academic_program_offer',
            'parent_columns': ['academic_program', 'tyear', 'tsem']
        }]
    }


def subplan_declarations(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['academic_plan', 'academic_program', 'department', 'student_token', 'academic_subplan']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'primary_keys': ['academic_plan', 'academic_program', 'tyear', 'tsem', 'degree', 'department',
                         'student_token', 'academic_subplan'],
        'foreign_keys': [{
            'columns': ['academic_plan', 'academic_program', 'tyear', 'tsem', 'degree', 'department', 'student_token'],
            'parent': 'student_enrolment'
        }, {
            'columns': ['academic_plan', 'academic_program', 'tyear', 'tsem', 'academic_subplan'],
            'parent': 'academic_subplan_offer',
        }]
    }


def career_enrolment(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['academic_plan', 'academic_program', 'department', 'student_token', 'academic_subplan']
    attributes = Table.learn_meta(src, id_cols, ['career_nbr'])
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'foreign_keys': [{
            'columns': ['academic_plan', 'academic_program', 'tyear', 'tsem', 'degree', 'department', 'student_token'],
            'parent': 'student_enrolment'
        }]
    }
