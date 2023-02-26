"""Table data processing."""

from .sis import personal_data, sis_academic_career, sis_academic_program, sis_academic_program_offer, sis_plan_offer, \
    sis_academic_plan, sis_enrolment, sis_milestone
from .uci_gym import uci_gym
from .module import module_offer, module_enrolment
from .series import wifi, luminus

__all__ = (
    'personal_data',
    'sis_academic_career',
    'sis_academic_program_offer',
    'sis_academic_program',
    'sis_plan_offer',
    'sis_academic_plan',
    'sis_enrolment',
    'sis_milestone',
    'uci_gym',
    'module_offer',
    'module_enrolment',
    'wifi',
    'luminus'
)
