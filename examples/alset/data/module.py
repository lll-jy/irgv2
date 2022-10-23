"""Module-related tables."""

import re

import pandas as pd

from .sis import _process_term


def module_offer(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Processed table**:

    Module information, including credits, grading basis, offer department, and level of instruction.
    """
    columns = ['module_code', 'module_credits', 'grading_basis']
    src = src[columns]
    extracted = src['module_code'].apply(
        lambda x: re.match('^[A-Z]+[^A-Z\d]*\d', x).group()
    )
    src.loc[:, 'offer_department'] = extracted.apply(
        lambda x: x[:-1]
    )
    src.loc[:, 'level'] = extracted.apply(
        lambda x: int(x[-1])
    )
    return src.drop_duplicates().reset_index(drop=True)


def module_enrolment(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Processed table**:

    Module enrolment, per student, per module, per term.
    """
    columns = ['student_token', 'module_code', 'academic_career', 'term', 'requirement_designation']
    src = src[columns]
    src = _process_term(src, '')
    return src
