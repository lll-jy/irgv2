"""Data processor for ALSET database."""

from typing import Dict, List, Optional
from types import FunctionType
import shutil
import os

import pandas as pd

from ..processor import DatabaseProcessor
from . import data, metadata


ALSET_PROCESSORS: Dict[str, FunctionType] = {
    'personal_data': data.personal_data,
    'sis_academic_career': data.sis_academic_career,
    'sis_academic_program_offer': data.sis_academic_program_offer,
    'sis_academic_program': data.sis_academic_program,
    'sis_plan_offer': data.sis_plan_offer,
    'sis_academic_plan': data.sis_academic_plan,
    'sis_enrolment': data.sis_enrolment,
    'gym': data.gym,
    'sis_module_enrolment': data.sis_module_enrolment,
    'sis_course': data.sis_course,
    'sis_credits': data.sis_credits,
    'sis_milestone': data.sis_milestone
}
"""ALSET table data processors."""

ALSET_META_CONSTRUCTORS: Dict[str, FunctionType] = {
    'personal_data': metadata.personal_data,
    'sis_academic_career': metadata.sis_academic_career,
    'sis_academic_program_offer': metadata.sis_academic_program_offer,
    'sis_academic_program': metadata.sis_academic_program,
    'sis_plan_offer': metadata.sis_plan_offer,
    'sis_academic_plan': metadata.sis_academic_plan,
    'sis_enrolment': metadata.sis_enrolment,
    'gym': metadata.gym,
    'sis_module_enrolment': metadata.sis_module_enrolment,
    'sis_course': metadata.sis_course,
    'sis_credits': metadata.sis_credits,
    'sis_milestone': metadata.sis_milestone
}
"""ALSET metadata constructors for each table."""

ALSET_PROCESS_NAME_MAP: Dict[str, str] = {
    'personal_data': 'sis/personal_data',
    'sis_academic_career': 'sis/program_enrolment',
    'sis_academic_program_offer': 'sis/program_enrolment',
    'sis_academic_program': 'sis/program_enrolment',
    'sis_plan_offer': 'sis/program_enrolment',
    'sis_academic_plan': 'sis/program_enrolment',
    'sis_enrolment': 'sis/program_enrolment',
    'gym': 'sis/gym',
    'sis_module_enrolment': 'sis/module_enrolment',
    'sis_course': 'sis/module_enrolment',
    'sis_credits': 'sis/module_enrolment',
    'sis_milestone': 'sis/milestone'
}
"""ALSET source data file names (without extension) for all tables."""


class ALSETProcessor(DatabaseProcessor):
    """Data processor for ALSET database."""
    def __init__(self, src_data_dir: str, data_dir: str, meta_dir: str, out: str, tables: Optional[List[str]] = None):
        """
        **Args**:

        Arguments to `DatabaseProcessor`.
        If tables is not specified, all recognized tables are processed.
        """
        if tables is None:
            tables = [*ALSET_PROCESSORS]
        super().__init__('alset', src_data_dir, data_dir, meta_dir, tables, out)

    @property
    def _table_data_processors(self) -> Dict[str, FunctionType]:
        return ALSET_PROCESSORS

    @property
    def _table_src_name_map(self) -> Dict[str, str]:
        return ALSET_PROCESS_NAME_MAP

    @property
    def _table_metadata_constructors(self) -> Dict[str, FunctionType]:
        return ALSET_META_CONSTRUCTORS

    @property
    def _source_encoding(self) -> Optional[str]:
        return None

    def postprocess(self, output_dir: Optional[str] = None, sample: Optional[int] = None):
        if sample is not None:
            if 'personal_data' not in self._tables:
                raise ValueError('Cannot sample ALSET database without personal data table.')
            if output_dir is None:
                raise ValueError('Cannot sample without specifying output directory.')
            os.makedirs(output_dir, exist_ok=True)
            personal_data = pd.read_pickle(os.path.join(self._data_dir, 'personal_data.pkl'))
            selected_students = personal_data['student_token'].sample(sample)
            for table_name in self._tables:
                table_data = pd.read_pickle(os.path.join(self._data_dir, f'{table_name}.pkl'))
                if 'student_token' not in set(table_data.columns):
                    table_data.to_pickle(os.path.join(output_dir, f'{table_name}.pkl'))
                else:
                    table_data = table_data[table_data['student_token'].isin(selected_students)].reset_index(drop=True)
                    table_data.to_pickle(os.path.join(output_dir, f'{table_name}.pkl'))
        elif output_dir is not None:
            shutil.copytree(self._data_dir, output_dir, dirs_exist_ok=True)
