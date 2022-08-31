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
        2011: 'gradtype',
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
        2016: 'conferred_degree_1st_major',
        2017: 'conferreddegreedescr',
        2018: 'conferreddegreedescr',
        2020: 'degreeconferred'
    },
    'major': {
        2010: 'major1',
        2011: 'major1_org',
        2012: 'major1',
        2013: 'major_1',
        2014: 'major1_1',
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
    },
    'faculty': {
        2010: 'faculty',
        2011: 'faculty_coded_org',
        2012: 'faculty',
        2013: 'college',
        2014: 'nus_faculty',
        2016: 'faculty_1st_major',
        2017: 'facultydescr',
        2018: 'facultydescr',
        2020: 'facultydescr'
    }
}
_GES_COLUMN_VALUE_EQUIV: Dict[str, Dict[Any, Set]] = {
    'academic_load': {
        'Full-Time': {'Full-Time', 'Full Time', '1', 'Full-Time2', 'Full-Time3'},
        'Part-Time': {'Part-Time', 'Part Time', '2'},
        'Follow-Up': {'Follow-Up', 'Follow Up', 'Follow up'}
    },
    'gender': {
        'M': {'M', 'Male', '1'},
        'F': {'F', 'Female', '2'}
    },
    'degree': {
        'Bachelor of Arts': {
            'B.A.', 'B.A. (Hons.)', 'Arts', 'Arts [Hons]', 'Bachelor of Arts', 'Bachelor of Arts (Hons)',
            'Bachelor of Arts (Hon)', 'Bachelor of Arts with Honours', '1', '2'
        },
        'Bachelor of Social Sciences': {
            'B.Soc.Sci. (Hons.)', 'Social Sciences [Hons]', 'Bachelor of Social Sciences (Hons)',
            'Bachelor of Social Sciences', '3', '4'
        },
        'Bachelor of Science': {
            'B.Sc. (Hons.)', 'B.Sc.', 'Science [Hons]', 'Science', 'Bachelor of Science (Hons)', 'Bachelor of Science',
            'Bachelor of Science (Hon)', 'Bachelor of Science with Honours', '17', '18', '20'
        },
        'Bachelor of Engineering (Electrical Engineering)': {
            'B.Eng. (Elect.)', 'Electrical Engineering', 'Bachelor of Engineering (Electrical Engineering)',
            'Bachelor of Engineering (Electric', '9'
        },
        'Bachelor of Business Administration': {
            'B.B.A.', 'B.B.A. (Hons.)', 'Business Administration [Honours]', 'Business Administration [3-yr programme]',
            'Bachelor of Business Administration (Hons)', 'Bachelor of Business Administration',
            'Bachelor of Business Administrati', 'Bachelor of Business Administration (Hon)', '22', '23'
        },
        'Bachelor of Engineering (Mechanical Engineering)': {
            'B.Eng. (Mech.)', 'Mechanical Engineering', 'Bachelor of Engineering (Mechanical Engineering)',
            'Bachelor of Engineering (Mechanic', '14'
        },
        'Bachelor of Engineering (Chemical Engineering)': {
            'B.Eng. (Chem.)', 'Chemical Engineering', 'Bachelor of Engineering (Chemical Engineering)',
            'Bachelor of Engineering (Chemical', '6'
        },
        'Bachelor of Laws': {
            'LL.B. (Hons.)', 'Law', 'Follow-Up: Law', 'Bachelor of Laws ', 'Bachelor of Laws (LLB) (Hons)', '15', '47'
        },
        'Bachelor of Science (Real Estate)': {
            'B.Sc. (Real Est.)', 'Real Estate', 'Bachelor of Science (Real Estate)', '36'
        },
        'Bachelor of Medicine and Bachelor of Surgery': {
            'M.B.B.S.', 'Medicine and Bachelor of Surgery', 'Follow-Up: Medicine and Bachelor of Surgery',
            'Bachelor of Medicine and Bachelor of Surgery', 'Bachelor of Medicine and Bachelor', '39', '48'
        },
        'Bachelor of Science (Project and Facilities Management)': {
            'B.Sc. (Proj. & Facilities Mgt.)', 'Project and Facilities Management',
            'Bachelor of Science (Project and Facilities Management)', 'Bachelor of Science (Project and',
            'Bachelor of Science (Project and Facilities Manag', 'B.Sc. (Bldg.)', '35'
        },
        'Bachelor of Science (Pharmacy)': {
            'B.Sc. (Pharm.) (Hons.)', 'Pharmacy', 'Follow-Up: Pharmacy', 'Bachelor of Science (Pharmacy) (Hons)',
            'Bachelor of Science (Pharmacy)', '21', '49'
        },
        'Bachelor of Technology (Electronics Engineering)': {
            'B. Tech. (Electronics Eng.)', 'Part-Time: B Tech [Electronics Eng]',
            'Bachelor of Technology (Electronics Engineering)', '42'
        },
        'Bachelor of Computing (Information Systems)': {
            'B.Comp. (I.S.)', 'Information Systems', 'Bachelor of Computing (Information Systems)',
            'Bachelor of Computing (Informatio', '32'
        },
        'Bachelor of Engineering (Computer Engineering)': {
            'B.Eng. (Comp.)', 'B.Comp. (Comp.Eng.)', 'Computer Engineering',
            'Bachelor of Engineering (Computer Engineering)', 'Bachelor of Engineering (Computer',
            'Bachelor of Computing (Computer Engineering)', '8', '29'
        },
        'Bachelor of Business Administration (Accountancy)': {
            'B.B.A. (Accountancy)', 'B.B.A. (Accountancy) (Hons.)', 'Business Administration [Accountancy]',
            'Business Administration [Accountancy] [Honours]', 'Bachelor of Business Administration (Accountancy)',
            'Bachelor of Business Administration (Accountancy) (Hons)',
            'Bachelor of Business Administration (Accountancy) (Hon)', '24', '25'
        },
        'Bachelor of Engineering (Civil Engineering)': {
            'B.Eng. (Civil)', 'Civil Engineering', 'Bachelor of Engineering (Civil Engineering)', '7'
        },
        'Bachelor of Engineering (Industrial and Systems Engineering)': {
            'B.Eng. (ISE.)', 'Industrial and Systems Engineering',
            'Bachelor of Engineering (Industrial and Systems Engineering)', 'Bachelor of Engineering (Industri',
            'Bachelor of Engineering (Industrial and Systems E', '12'
        },
        'Bachelor of Engineering (Biomedical Engineering)': {
            'B.Eng. (Bioengineering)', 'Bioengineering', 'Bachelor of Engineering (Bioengineering)',
            'Bachelor of Engineering (Biomedical Engineering)', 'Bachelor of Engineering (Biomedic', '5'
        },
        'Bachelor of Computing (Electronic Commerce)': {
            'B.Comp. (E.Commerce)', 'Electronic Commerce', 'Bachelor of Computing (Electronic Commerce)',
            'Bachelor of Computing (Electronic', '31'
        },
        'Bachelor of Technology (Mechanical Engineering)': {
            'B. Tech. (Mech. Eng.)', 'Part-Time: B Tech [Mech Eng]', 'Bachelor of Technology (Mechanical Engineering)',
            '45'
        },
        'Bachelor of Computing (Computer Science)': {
            'B.Comp. (Comp.Science)', 'Computer Science', 'Bachelor of Computing (Computer Science)',
            'Bachelor of Computing (Computer S', '30'
        },
        'Bachelor of Applied Science': {
            'B.Appl.Sci. (Hons.)', 'B.Appl.Sci.', 'Applied Science [Hons]', 'Applied Science',
            'Bachelor of Applied Science (Hons)', 'Bachelor of Applied Science', 'Bachelor of Applied Science (Hon)',
            '16'
        },
        'Bachelor of Computing (Communications and Media)': {
            'B.Comp. (Comm. and Media)', 'Communications and Media', 'Bachelor of Computing (Communications and Media)',
            '27'
        },
        'Bachelor of Science (Nursing)': {
            'B.Sc. (Nursing)', 'Nursing [Honours]', 'Nursing', 'Bachelor of Science (Nursing)',
            'Bachelor of Science (Nursing) (Hons)', 'Bachelor of Science (Nursing) (Hon)',
            'Bachelor of Science (Nursing) (Ho', '37', '38'
        },
        'Bachelor of Dental Surgery': {
            'B.D.S.', 'Dental Surgery', 'Bachelor of Dental Surgery'
        },
        'Bachelor of Engineering (Materials Science and Engineering)': {
            'B.Eng. (Materials Sci. & Eng.)', 'Materials Science and Engineering',
            'Bachelor of Engineering (Materials Science and Engineering)', 'Bachelor of Engineering (Material',
            'Bachelor of Engineering (Materials Science and En', '13'
        },
        'Bachelor of Engineering (Environmental Engineering)': {
            'B.Eng. (Environ.)', 'Environmental Engineering', 'Bachelor of Engineering (Environmental Engineering)',
            'Bachelor of Engineering (Environm', 'Bachelor of Engineering (Environmental Engineerin', '11'
        },
        'Bachelor of Music': {
            'B.Mus.', 'Music', 'Bachelor of Music', '40'
        },
        'Bachelor of Engineering (Engineering Science)': {
            'B.Eng. (Eng. Science)', 'Engineering Science', 'Bachelor of Engineering (Engineering Science)',
            'Bachelor of Engineering (Engineer', '10'
        },
        'Bachelor of Arts (Industrial Design)': {
            'B.A. (ID.)', 'Industrial Design', 'Bachelor of Arts (Industrial Design)',
            'Bachelor of Arts (Industrial Desi', '34'
        },
        'Bachelor of Computing': {
            'B.Comp.', 'Computing', '26'
        },
        'Bachelor of Arts (Architecture)': {
            'B.A. (Arch.)', 'Architecture', 'Follow-Up: Architecture', 'Bachelor of Arts (Architecture)', '33'
        },
        'Bachelor of Science (Computational Biology)': {
            'B.Sc. (Comp.Bio.)', 'Computational Biology', 'Bachelor of Science (Computational Biology)',
            'Bachelor of Computing (Computational Biology)', 'Bachelor of Science (Computationa',
            'Bachelor of Computing (Computatio', '19', '28'
        },
        'Bachelor of Technology (Chemical Engineering)': {
            'B. Tech. (Chem. Eng.)', 'Part-Time: B Tech [Chem Eng]', 'Bachelor of Technology (Chemical Engineering)',
            '41'
        },
        'Bachelor of Technology (Manufacturing Engineering)': {
            'B. Tech. (Manufacturing Eng.)', 'Bachelor of Technology (Manufacturing Engineering)', '46'
        },
        'Bachelor of Technology (Industrial and Management Engineering)': {
            'Part-Time: B Tech [Industrial and Management Eng]',
            'Bachelor of Technology (Industrial and Management Engineering)',
            'Bachelor of Technology (Industrial and Management Engineerin', '43'
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
        'BBA': {'BBA', 'BBA (HONS)'},
        'Economics': {'Economics (Major)', 'ECONOMICS', 'ECONOMICS (HONS)'},
        'Life Sciences': {'Life Scences (Major)', 'LIFE SCIENCES', 'LIFE SCIENCES (HONS)'},
        'Communications & New Media': {'Comms & New Media (Major)', 'Comms and New Media (Major)', 'COMMS & NEW MEDIA',
                                       'COMMS & NEW MEDIA (HONS)'},
        'Chemistry': {'Chemistry (Major)', 'CHEMISTRY', 'CHEMISTRY (HONS)'},
        'Applied Chemistry': {'Applied Chemistry (Major)', 'APPLIED CHEMISTRY'},
        'Psychology': {'Psychology (Major)', 'PSYCHOLOGY', 'PSYCHOLOGY (HONS)'},
        'Accountancy': {'Accountancy', 'ACCOUNTANCY', 'ACCOUNTANCY (HONS)'},
        'Sociology': {'Sociology (Major)', 'SOCIOLOGY', 'SOCIOLOGY (HONS)'},
        'Mathematics': {'Mathematics (Major)', 'MATHEMATICS', 'MATHEMATICS (HONS)'},
        'Applied Mathematics': {'Applied Mathematics (Major)', 'APPLIED MATHEMATICS', 'APPLIED MATHEMATICS (HONS)'},
        'Statistics': {'Statistics (Major)', 'STATISTICS', 'STATISTICS (HONS)'},
        'Real Estate': {'REAL ESTATE', 'REAL ESTATE (HONS)'},
        'Political Science': {'Political Science (Major)', 'POLITICAL SCIENCE', 'POLITICAL SCIENCE (HONS)'},
        'Nursing': {'Nursing (Major)', 'NURSING', 'NURSING (HONS)', 'Nursing (Major) PCP'},
        'Geography': {'Geography (Major)', 'GEOGRAPHY', 'GEOGRAPHY (HONS)'},
        'Env-Geo': {'ENVIRONMENTAL STUDIES-GEO HONS'},
        'Env-Bio': {'ENVIRONMENTAL STUDIES-BIO HONS'},
        'Physics': {'Physics (Major)', 'PHYSICS', 'PHYSICS (HONS)'},
        'Business': {'Business (Major)', 'BUSINESS'},
        'Food Science & Tech': {'Food Science & Tech (Major)', 'FOOD SCIENCE & TECH', 'Food Science and Tech (Major)',
                                'FOOD SCIENCE & TECH (HONS)', 'Food Sci & Tech (Major)'},
        'History': {'History (Major)', 'HISTORY', 'HISTORY (HONS)'},
        'English Literature': {'English Lit (Major)', 'ENGLISH LIT', 'ENGLISH LIT (HONS)'},
        'English Language': {'English Language (Major)', 'ENGLISH LANGUAGE', 'ENGLISH LANGUAGE (HONS)'},
        'Social Work': {'Social Work (Major)', 'SOCIAL WORK', 'SOCIAL WORK (HONS)'},
        'Japanese Studies': {'Japanese Studies (Major)', 'JAPANESE STUDIES', 'JAPANESE STUDIES (HONS)'},
        'Chinese Language': {'Chinese Language (Major)', 'CHINESE LANGUAGE'},
        'Chinese Studies': {'Chinese Studies (Major)', 'CHINESE STUDIES', 'CHINESE STUDIES (HON-CL TRACK)',
                            'CHINESE STUDIES (HONS)'},
        'Technology': {'Technology (Major)'},
        'Theatre Studies': {'Theatre Studies (Major)', 'THEATRE STUDIES', 'THEATRE STUDIES (HONS)'},
        'Malay Studies': {'Malay Studies (Major)', 'MALAY STUDIES', 'MALAY STUDIES (HONS)'},
        'European Studies': {'European Studies (Major)', 'EUROPEAN STUDIES', 'EUROPEAN STUDIES (HONS)'},
        'Philosophy': {'Philosophy (Major)', 'PHILOSOPHY', 'PHILOSOPHY (HONS)'},
        'S.E. Asian Studies': {'S.E. Asian Studies (Major)', 'S.E. ASIAN STUDIES', 'S.E. ASIAN STUDIES (HONS)'},
        'South Asian Studies': {'SOUTH ASIAN STUD', 'South Asian Stud (Major)'},
        'Communications & Media': {'COMMUNICATIONS & MEDIA', 'COMMUNICATIONS & MEDIA (HONS)'},
        'E-Commerce': {'ELECTRONIC COMMERCE', 'ELECTRONIC COMMERCE (HONS)'},
        'Medicine': {'MEDICINE', 'Medicine (Major)'},
        'Law': {'LAW', 'LAW (HONS) - 4 YEAR', 'LAW (HONS) - 3 YEAR'},
        'Mechanical Engineering': {'MECHANICAL ENGINEERING', 'MECHANICAL ENGINEERING (HONS)'},
        'Chemical Engineering': {'CHEMICAL ENGINEERING', 'CHEMICAL ENGINEERING (HONS)'},
        'Electrical Engineering': {'ELECTRICAL ENGINEERING', 'ELECTRICAL ENGINEERING (HONS)'},
        'Electronics Engineering': {'ELECTRONICS ENGINEERING', 'ELECTRONICS ENG (HONS)'},
        'Pharmacy': {'PHARMACY', 'PHARMACY (HONS)'},
        'Computer Engineering': {'COMPUTER ENGINEERING', 'COMPUTER ENGINEERING-CEG', 'COMPUTER ENGINEERING(HONS)-CEG'},
        'Computer Science': {'COMPUTER SCIENCE', 'COMPUTER SCIENCE (HONS)'},
        'Computational Biology': {'COMPUTATIONAL BIOLOGY', 'COMPUTATIONAL BIOLOGY (HONS)'},
        'Composition': {'COMPOSITION'},
        'Architecture': {'ARCHITECTURE', 'ARCHITECTURE (HONS)'},
        'Project & Facilities Management': {'PROJECT & FAC. MGT', 'PROJECT & FAC. MGT (HONS)'},
        'Information Systems': {'INFORMATION SYSTEMS', 'INFORMATION SYSTEMS (HONS)'},
        'Industrial & System Engineering': {'INDUSTRIAL & SYS ENG', 'INDUSTRIAL AND SYSTEMS ENGINEERING',
                                            'INDUSTRIAL & SYS ENG (HONS)'},
        'Industrial & Management Engineering': {'INDUSTRIAL AND MANAGEMENT ENGINEERING', 'INDUSTRIAL & MGT ENG (HONS)'},
        'Industrial Design': {'INDUSTRIAL DESIGN', 'INDUSTRIAL DESIGN (HONS)'},
        'Civil Engineering': {'CIVIL ENGINEERING', 'CIVIL ENGINEERING (HONS)'},
        'Biomedical Engineering': {'BIOENGINEERING', 'BIOMEDICAL ENGINEERING', 'BIOMEDICAL ENGINEERING (HONS)'},
        'Environmental Engineering': {'ENVIRONMENTAL ENG', 'ENVIRONMENTAL ENG (HONS)'},
        'Engineering Science': {'ENGINEERING SCIENCE', 'ENGINEERING SCIENCE (HONS)'},
        'Materials Science & Engineering': {'MATERIALS SC & ENG', 'MATERIALS SCIENCE AND ENGINEERING',
                                            'MATERIALS SC & ENG (HONS)'},
        'Dental Surgery': {'Dental Surgery (Major)', 'DENTISTRY'},
        'Quantitative Finance': {'Quantitative Finance (Major)', 'QUANTITATIVE FINANCE', 'QUANTITATIVE FINANCE (HONS)',
                                 'Quantitative Finance (Major'},
        'Violin Performance': {'VIOLIN PERFORMANCE', 'VIOLIN PERFORMANCE (HONS)'},
        'Cello Performance': {'CELLO PERFORMANCE', 'CELLO PERFORMANCE (HONS)'},
        'French Horn Performance': {'FRENCH HORN PERFORMANCE', 'FRENCH HORN PERFORMANCE (HONS)'},
        'Flute Performance': {'FLUTE PERFORMANCE', 'FLUTE PERFORMANCE (HONS)'},
        'Double Bass Performance': {'DOUBLE BASS PERFORMANCE', 'DOUBLE BASS PERFORMANCE (HONS)'},
        'Trumpet Performance': {'TRUMPET PERFORMANCE', 'TRUMPET PERFORMANCE (HONS)'},
        'Harp Performance': {'HARP PERFORMANCE', 'HARP PERFORMANCE (HONS)'},
        'Percussion Performance': {'PERCUSSION PERFORMANCE', 'PERCUSSION PERFORMANCE (HONS)'},
        'Piano Performance': {'PIANO PERFORMANCE', 'PIANO PERFORMANCE (HONS)'},
        'Clarinet Performance': {'CLARINET PERFORMANCE', 'CLARINET PERFORMANCE (HONS)'},
        'Tuba Performance': {'TUBA PERFORMANCE'},
        'Viola Performance': {'VIOLA PERFORMANCE', 'VIOLA PERFORMANCE (HONS)'},
        'Trombone Performance': {'TROMBONE PERFORMANCE', 'TROMBONE PERFORMANCE (HONS)'},
        'Bassoon Performance': {'BASSOON PERFORMANCE'},
        'Oboe Performance': {'OBOE PERFORMANCE (HONS)'},
        'Global Studies': {'Global Studies (Major)', 'GLOBAL STUDIES (HONS)'},
        'Voice': {'VOICE', 'VOICE (HONS)'},
        'Act Studies & Economics': {'ACT ST. & ECONS'},
        'Recording Arts & Science': {'RECORDING ARTS & SCI.', 'RECORDING ARTS & SCI. (HONS)'}
    },
    'usp': {
        'Y': {'1', '3', '4', 'Yes'},
        'N': {'2', '.', 'No'}
    },
    'faculty': {
        'Faculty of Engineering': {'Faculty of Engineering', '3', 'FACULTY OF ENGINEERING'},
        'Faculty of Arts & Social Sciences': {'Faculty of Arts & Social Sci', '1', 'Arts & Social Sciences',
                                              'Faculty of Arts & Social Science', 'Faculty of Arts & Social Sciences',
                                              'FACULTY OF ARTS & SOCIAL SCI'},
        'Faculty of Science': {'Faculty of Science', '5', 'FACULTY OF SCIENCE'},
        'NUS Business School': {'NUS Business School', '6', 'NUS BUSINESS SCHOOL'},
        'Faculty of Law': {'Faculty of Law', '4', 'FACULTY OF LAW'},
        'School of Cont & Lifelong Edun': {'School of Cont & Lifelong Edun', '8'},
        'School of Computing': {'School of Computing', '7', 'SCHOOL OF COMPUTING'},
        'YLL School of Medicine': {'YLL School of Medicine', '9', 'YONG LOO LIN SCHOOL (MEDICINE)',
                                   'Yong Loo Lin School (Medicine)', 'Yong LOo Lin Sch of Medicine'},
        'School of Design & Environment': {'School of Design & Environment', '2', 'School of Design and Environment',
                                           'SCHOOL OF DESIGN & ENVIRONMENT'},
        'Faculty of Dentistry': {'Faculty of Dentistry', 'FACULTY OF DENTISTRY'},
        'YST Conservatory of Music': {'YST Conservatory of Music', '10', 'Yong Siew Toh Conservatory of Music',
                                      'YST CONSERVATORY OF MUSIC'},
        'Multi Disciplinary Programme': {'Multi Disciplinary Programme', 'MULTI DISCIPLINARY PROGRAMME',
                                         'Multidisciplinary Programme'},
        'Yale-NUS College': {'Yale-NUS College'}
    }
}
_GES_COLUMN_DTYPES: Dict[str, Union[Type, str]] = {
    'student_token': str,
    'academic_load': str,
    'gender': str,
    'degree': str,
    'major': str,
    'usp': str,
    'faculty': str
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


def _process_year(year: int, df: pd.DataFrame) -> pd.DataFrame:
    result = df.rename(columns={
        col: col_by_year[year]
        for col, col_by_year in _GES_COLUMNS.items()
        if year in col_by_year
    }).astype(_GES_COLUMN_DTYPES)
    result.loc[:, 'usp'] = result['usp'].fillna('N')
    for col in result.columns:
        if _GES_COLUMN_DTYPES[col] == str:
            result.loc[:, col] = result[col].apply(lambda x: x if pd.isnull(x) else
                                                   np.nan if x.strip() == '' else x.strip())
        result.loc[:, col] = result[col].apply(partial(_get_canonical_value, col=col))
    result.loc[:, 'major'] = result.apply(
        lambda row: row['major'] if not pd.isnull(row['major']) and row['major'] != 'Nil'
        else _degree2major_map[row['degree']] if row['degree'] in _degree2major_map else np.nan, axis=1
    )
    return result


def ges2010(src: pd.DataFrame) -> pd.DataFrame:
    return _process_year(2010, src)


def ges2015(src: pd.DataFrame) -> pd.DataFrame:
    result = src.copy()
    result.loc[:, 'degree'] = _combine_columns('nus_programme', 'combineprogramme', src)
    result.loc[:, 'major'] = src.apply(
        lambda row: row['major1_1'] if not pd.isnull(row['major1_1'])
        and row['major1_1'].strip() != '' else row['major'], axis=1
    )
    result.loc[:, 'faculty'] = src.apply(
        lambda row: row['nus_faculty'] if not pd.isnull(row['nus_faculty'])
        else row['combineprogramme']
    )
    return _process_year(2015, result)


def ges2019(src: pd.DataFrame) -> pd.DataFrame:
    result = src.copy()
    result.loc[:, 'degree'] = _combine_columns('conferreddegreedescr', 'conferreddegreedescr1', src)
    result.loc[:, 'usp'] = _combine_columns('uspscholaryorn', 'usp', src)
    result.loc[:, 'faculty'] = _combine_columns('faculty', 'facultydescr', src)
    return _process_year(2019, result)
