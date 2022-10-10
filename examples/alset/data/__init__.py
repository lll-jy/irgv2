"""Table data processing."""

from .sis import personal_data, sis_academic_career, sis_academic_program, sis_academic_program_offer, sis_plan_offer, \
    sis_academic_plan, sis_enrolment, gym, sis_module_enrolment, sis_course, sis_credits, sis_milestone

__all__ = (
    'personal_data',
    'sis_academic_career',
    'sis_academic_program_offer',
    'sis_academic_program',
    'sis_plan_offer',
    'sis_academic_plan',
    'sis_enrolment',
    'gym',
    'sis_module_enrolment',
    'sis_course',
    'sis_credits',
    'sis_milestone'
)
