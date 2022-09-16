"""Student Information System tables."""

from typing import Dict, Any

import pandas as pd

from irg.schema import Table


def personal_data(src: pd.DataFrame) -> Dict[str, Any]:
    pass


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