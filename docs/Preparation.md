# Pre-defined Datasets

Supported pre-defined datasets are declared in [examples](../examples).

Usage of the processing code for pre-defined datasets can be found in [`UserGuide`](./user_guide#data-preparation).

# Custom Datasets

## Create Examples Package

1. In [examples](../examples) package, create a new package for the custom dataset.
1. In the new package, create two sub-packages named `data` and `metadata` respectively.
1. In `data` package, create some files with data processing code.
    - The files can be ordered in arbitrary manner for the users' preference.
      It is suggested to group tables on relevant topics semantically together.
    - For each table, in one of the files mentioned above, declare a data processor as a function that takes in the 
      source dataframe and outputs the processed dataframe.
      For more details on recommendations of how to do processing, please see [next section](#processing-data).
    - In this package (`__init__.py`), export the processors of all tables.
1. In `metadata` package, create files in similar manner to `data`.
    - For each table, in one of the files, declare a metadata constructor as a function that takes in the processed
      dataframe and outputs metadata as a `dict` of the table.
      For more details on recommendations of how to do processing, please see 
      [next next section](#constructing-metadata).
    - In this package (`__init__.py`), export the metadata constructors of all tables.
1. In the package handling this custom database, construct a dictionary mapping table names to data processors, and 
   another dictionary mapping table names to metadata constructors. Export these two dictionaries.
   The following is an example of `__init__.py` for any custom database package in `example`. 
        ```python
        from typing import Dict
        from types import FunctionType
        
        from . import data, metadata
        
        __all__ = (
           'MY_NEW_PROCESSORS',
           'MY_NEW_META_CONSTRUCTORS'
        )
        
        MY_NEW_PROCESSORS: Dict[str, FunctionType] = {
           'my_table1': data.my_table1,
           'my_table2': data.my_table2
        }
        
        MY_NEW_META_CONSTRUCTORS: Dict[str, FunctionType] = {
           'my_table1': metadata.table1,
           'my_table2': metadata.table2
        }
        ```

1. In `examples/__init__.py`, import the processors dictionary and metadata constructors dictionary
   for the new database and add in the database-level dictionary for processors and metadata constructors.
   (Add one element in `PROCESSORS` and `META_CONSTRUCTORS` respectively.)

## Processing Data

1. Identify needed columns from the source data and extract them.
1. Identify categorical columns represented as numbers (for example, label IDs) and convert them to `Int32` type 
   (`NaN`-tolerant integer type).
1. Do other custom processing if needed.
1. It is highly suggested to break big tables that semantically contain multiple tables into different tables.
   To check which columns are controlled under which other columns as primary key, one can use the function below.
   `pk_cols` are the primary keys to be assumed, and `cond` are columns that are conditioned on (that is, if column A
   is uniquely determined by `pk_cols` under each value or combination values of `cond` columns, the column is still 
   considered) following the primary key set. The returned result is a set of column names that are uniquely determined
   by `pk_cols` as described, which can be used to construct a table with `pk_cols` as primary keys.
   
        def find_cols_follow_pk(pk_cols, dropna=False, cond=None):
            valid_cols = set(df.columns)
            if cond is None:
                for _, data in df.groupby(by=pk_cols):
                    cols_to_rm = set()
                    for col in valid_cols:
                        if data[col].nunique(dropna=dropna) > 1:
                            cols_to_rm.add(col)
                    valid_cols -= cols_to_rm
            else:
                for _, df_data in df.groupby(by=cond):
                    for _, data in df_data.groupby(by=pk_cols):
                        cols_to_rm = set()
                        for col in valid_cols:
                            if data[col].nunique(dropna=dropna) > 1:
                                cols_to_rm.add(col)
                        valid_cols -= cols_to_rm
            return valid_cols

1. Some tables represent some child table information flattened if the degree is bounded to a small value.
   For example, for each student, in some school, each student can be enrolled in at most two programs, so 
   the table of program enrolment has columns `student_id`, `program1` and `program2`, possibly followed by
   a set of other columns describing each program, like `program_type1`, `program_type`, `program_faculty1`, 
   `program_faculty2`. And if a student is enrolled to one program only, all columns to `program2` are left as `NaN`.
   In such cases, it is highly suggested to reformat the table to each program enrolment occupying one row,
   which is more intuitive in the sense of relational database.
   To handle such data, one can use the following template (the template is for the example mentioned).
   
        def my_table(src: pd.DataFrame) -> pd.DataFrame:
            base_cols = ['program#', 'program_type#', 'program_faculty#']
            num_cat_cols = ['program_type']
            relevant_data = []
            shared_cols = ['student_id']
            for i in range(1, 3):
                relevant_group = src[shared_cols + [col.replace('#', f'{i}') for col in base_cols]]
                for col in num_cat_cols:
                    relevant_group[col] = relevant_group[col].astype('Int32')
                relevant_data.append(relevant_group)
            return pd.concat(relevant_data).dropna().reset_index(drop=True)

## Constructing Metadata

1. Identify ID columns (including primary and foreign keys) in a list.
1. Identify categorical columns represented as numbers (for example, label IDs).
1. Use the above information together with the processed dataframe as input to learn attribute meta.
1. Determine determinants and/or formulas if needed, based on understanding of table data.
1. Make custom changes if needed.

A complete example is shown as follows

```python
from typing import Dict, Any

import pandas as pd

from irg.schema import Table


def my_table(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['my_id']
    num_cat_cols = ['type_id']
    attributes = Table.learn_meta(src, id_cols, num_cat_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'determinants': [
            ['type_id', 'type_name']
        ],
        'primary_keys': id_cols,
        'foreign_keys': [
            {
                'columns': ['fk'],
                'parent': 'my_parent',
                'parent_columns': ['id']
            }
        ]
    }
```
