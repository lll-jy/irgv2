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
    'sis_milestone': data.sis_milestone,
    'uci_gym': data.uci_gym,
    'module_offer': data.module_offer,
    'module_enrolment': data.module_enrolment
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
    'sis_milestone': metadata.sis_milestone,
    'uci_gym': metadata.uci_gym,
    'module_offer': metadata.module_offer,
    'module_enrolment': metadata.module_enrolment
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
    'sis_milestone': 'sis/milestone',
    'uci_gym': 'sis/uci_gym',
    'module_offer': 'sis/module_enrolment',
    'module_enrolment': 'sis/module_enrolment'
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
            if os.path.exists(os.path.join(self._data_dir, 'module_offer.pkl')):
                module_offer = pd.read_pickle(os.path.join(self._data_dir, 'module_offer.pkl'))
                selected_modules = module_offer['module_code'].sample(sample)
            for table_name in self._tables:
                table_data = pd.read_pickle(os.path.join(self._data_dir, f'{table_name}.pkl'))
                columns = set(table_data.columns)
                if 'student_token' in columns:
                    table_data = table_data[table_data['student_token'].isin(selected_students)].reset_index(drop=True)
                if 'module_code' in columns:
                    table_data = table_data[table_data['module_code'].isin(selected_modules)].reset_index(drop=True)
                table_data.to_pickle(os.path.join(output_dir, f'{table_name}.pkl'))

        elif output_dir is not None:
            shutil.copytree(self._data_dir, output_dir, dirs_exist_ok=True)
