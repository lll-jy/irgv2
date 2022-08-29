"""Student Information System tables."""

from datetime import datetime
import re

import numpy as np
import pandas as pd


def personal_data(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Processed table**:
    """
    pass


_code2descr = {
    '00': '',
    '10': 'Semester 1',
    '20': 'Semester 2',
    '30': 'Special Term (Part1)',
    '40': 'Special Term (Part2)',
    '13': 'SMA Semester 1',
    '14': 'SMA Semseter 2',
    '23': 'SMA Semester 3',
    '11': 'Quarter 1',
    '12': 'Quarter 2',
    '21': 'Quarter 3',
    '22': 'Quarter 4',
    '15': 'August',
    '16': 'September',
    '17': 'October',
    '18': 'November',
    '19': 'December',
    '26': 'February',
    '27': 'March',
    '28': 'April',
    '31': 'May'
}

_early_terms = {
    3: '1997/1998 Semester 2',
    4: '1998/1999 Semester 1',
    5: '1998/1999 Semester 2',
    6: '1999/2000 Semester 1',
    7: '1999/2000 Semester 2'
}


def _term_code2descr(code: int) -> str:
    year = code // 100 + 2000
    if code < 10:
        return _early_terms[code]
    sem = _code2descr[str(code % 100).zfill(2)]
    return f'{year}/{year+1} {sem}'


def _process_term(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if f'{prefix}term_descr' in df:
        term = df[[f'{prefix}term', f'{prefix}term_descr']]
        df = df.drop(columns=[f'{prefix}term', f'{prefix}term_descr'])
    else:
        term = df[[f'{prefix}term']]
        term[f'{prefix}term_descr'] = term[f'{prefix}term'].apply(_term_code2descr)
        df = df.drop(columns=[f'{prefix}term'])
    year = term[f'{prefix}term_descr'].apply(lambda x: datetime(year=int(x[:4]), month=1, day=1))\
        .astype('datetime64[ns]')
    sem = term.apply(lambda row:
                     _code2descr[str(row[f'{prefix}term'] % 100).zfill(2)] if row[f'{prefix}term'] > 100
                     else re.split(r'(?<=^\d{4}/\d{4}) (?=.*$)', row[f'{prefix}term_descr'] + ' ')[1], axis=1)
    df[f'{prefix}tyear'] = year
    df[f'{prefix}tsem'] = sem
    return df


def academic_program_offer(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Raw table**:

    EduRec data.

    **Processed table**:

    Academic program offered each term.
    """
    result = pd.concat([
        src[['academic_program', 'academic_program_descr', 'term']].drop_duplicates().reset_index(drop=True),
        src[['dual_academic_program', 'dual_academic_program_descr', 'term']].rename(columns={
            'dual_academic_program': 'academic_program',
            'dual_academic_program_descr': 'academic_program_descr'
        }).drop_duplicates().reset_index(drop=True),
    ]).drop_duplicates().reset_index(drop=True)
    result = _process_term(result, '')
    return result


def academic_plan_offer(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Raw table**:

    EduRec data.

    **Processed table**:

    Academic plan offered each term.
    """
    result = src[['academic_career', 'academic_plan', 'academic_plan_descr', 
                  'academic_plan_type', 'academic_plan_type_descr',
                  'academic_program', 'term']]\
        .drop_duplicates().reset_index(drop=True)
    result = _process_term(result, '')
    return result


def academic_subplan_offer(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Raw table**:

    EduRec data.

    **Processed table**:

    Academic subplans offered each term per academic plan.
    """
    result = pd.concat([
        src[[
            f'academic_subplan{i}', f'academic_subplan{i}_descr',
            'academic_plan', 'academic_program', 'term'
        ]].drop_duplicates().reset_index(drop=True).rename(columns={
            f'academic_subplan{i}': 'academic_subplan',
            f'academic_subplan{i}_descr': 'academic_subplan_descr'
        })
        for i in range(1, 4)
    ]).drop_duplicates().reset_index(drop=True)
    result = _process_term(result, '')
    return result


def degree_offer(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Raw table**:

    EduRec data.

    **Processed table**:

    Degree programs offered each term.
    """
    result = src[['academic_plan', 'academic_program', 'term',
                  'degree', 'faculty_code', 'faculty_descr', 'department', 'department_descr']]\
        .drop_duplicates().reset_index(drop=True).astype({'faculty_code': 'Int32'})
    result = _process_term(result, '')
    return result


def student_enrolment(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Raw table**:

    EduRec data.

    **Processed table**:

    Enrolment per term per student, including academic career, plans, and degree information.
    """
    empty_str = ['attached_to_ri', 'attached_to_ri_descr', 'course_code', 'exchange_level',
                 'partner_university', 'partner_university_descr']
    result = src.rename(columns={'partner_unversity': 'partner_university'})[[
        'academic_plan', 'academic_program', 'term', 'degree', 'department', 'student_token',
        'academic_load_descr', 'attached_to_ri', 'attached_to_ri_descr', 'businessdate', 'course_code',
        'dual_academic_program', 'exchange_level', 'final_candidature_extension',
        'form_of_study', 'form_of_study_descr', 'partner_university', 'partner_university_descr'
    ]].drop_duplicates().reset_index(drop=True)
    result['businessdate'] = result['businessdate'].astype(str).astype('datetime64[ns]')
    for col in empty_str:
        result[col] = result[col].apply(lambda x: x if x.strip() != '' else np.nan)
    result = _process_term(result, '')
    return result


def subplan_declarations(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Raw table**:

    EduRec data.

    **Processed table**:

    Subplans per
    """


def academic_career(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Raw table**:

    EduRec data.

    **Processed table**:

    Academic career enrolment per student. Including information like important dates associated with the academic
    career, faculty, and department.
    """
    datetime_columns = ['businessdate', 'candidature_end_date', 'candidature_start_date', 'registration_date']
    int_columns = ['career_nbr']
    term_prefixes = ['admit_', 'completion_', 'expected_graduation_', 'requirement_']
    empty_str = ['attached_to_ri', 'attached_to_ri_descr']
    result = src[[
        'student_token', 'academic_career', 'academic_load_descr', 'admit_term', 'admit_term_descr',
        'attached_to_ri', 'attached_to_ri_descr', 'businessdate', 'candidature_end_date', 'candidature_start_date',
        'career_nbr', 'completion_term',
        'degree', 'degree_checkout_status', 'degree_checkout_status_descr', 'degree_descr',
        'department', 'department_descr', 'expected_graduation_term',
        'faculty_code', 'faculty_descr', 'final_candidature_extension',
        'registration_date', 'requirement_term', 'requirement_term_descr'
    ]].drop_duplicates().reset_index(drop=True) \
        .astype({col: str for col in datetime_columns}) \
        .astype({col: 'datetime64[ns]' for col in datetime_columns}
                | {col: 'Int32' for col in int_columns} | {'faculty_code': str})
    for prefix in term_prefixes:
        result = _process_term(result, prefix)
    for col in empty_str:
        result[col] = result[col].apply(lambda x: x if x.strip() != '' else np.nan)
    return result


def academic_program(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Raw table**:

    EduRec data.

    **Processed table**:

    Academic program enrolment per student.
    """
    result = src[[
        'student_token', 'academic_career', 'career_nbr',
        'academic_program', 'academic_program_descr', 'program_category',
        'dual_academic_program', 'dual_academic_program_descr',
        'exchange_level', 'partner_university_descr', 'partner_unversity'
    ]].rename(columns={'partner_unversity': 'partner_university'}).astype({'career_nbr': 'Int32'})
    result['exchange_level'] = result['exchange_level'].apply(lambda x: x.replace('( UG)', '(UG)'))
    empty_str = ['dual_academic_program', 'dual_academic_program_descr',
                 'exchange_level', 'partner_university_descr', 'partner_unversity']
    for col in empty_str:
        result[col] = result[col].apply(lambda x: x if x.strip() != '' else np.nan)
    return result


def academic_plan(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Raw table**:

    EduRec data.

    **Processed table**:

    Academic plan enrolment per student.
    """
    pass


def academic_subplan(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Raw table**:

    EduRec data.

    **Processed table**:

    Academic sub plans per student.
    """
    pass
