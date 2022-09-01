from functools import partial
from typing import Dict, Set, Any, Type, Union

import numpy as np
import pandas as pd


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
    }
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
    }
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
    'employment_status': 'string'
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
    for col in ['not_ft_perm_reason', 'employment_status']:
        result.loc[:, col] = result[col].replace({'Nil': np.nan})
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
    return _process_year(2012, result)


def ges2013(src: pd.DataFrame) -> pd.DataFrame:
    result = src.copy()
    return _process_year(2013, result)


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
    return _process_year(2015, result)


def ges2016(src: pd.DataFrame) -> pd.DataFrame:
    result = src.copy()
    result.loc[:, 'hon'] = result['honours_class_1st_major'].apply(
        lambda x: 'Y' if 'Hon' in x else 'N'
    )
    return _process_year(2016, result)


def ges2017(src: pd.DataFrame) -> pd.DataFrame:
    result = src.copy()
    return _process_year(2017, result)


def ges2018(src: pd.DataFrame) -> pd.DataFrame:
    result = src.copy()
    result.loc[:, 'noc'] = result['noc'].fillna('N')
    result.loc[:, 'hon'] = result['honours_descr'].apply(
        lambda x: 'Y' if 'Hon' in x else 'N'
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
    return _process_year(2019, result)


def ges2020(src: pd.DataFrame) -> pd.DataFrame:
    result = src.copy()
    result.loc[:, 'hon'] = src['honoursplandescr'].apply(
        lambda x: 'Y' if x.strip() != '' else 'N'
    )
    return _process_year(2020, result)
