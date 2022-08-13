"""
Global Terrorism Database processing.

Data is downloaded from [Kaggle](https://www.kaggle.com/datasets/START-UMD/gtd?resource=download).

The Global Terrorism Database (GTD) is an open-source database including information on terrorist attacks around the
world from 1970 through 2017. The GTD includes systematic data on domestic as well as international terrorist incidents
that have occurred during this time period and now includes more than 180,000 attacks. The database is maintained by
researchers at the National Consortium for the Study of Terrorism and Responses to Terrorism (START), headquartered at
the University of Maryland.
"""

from typing import Dict
from types import FunctionType

from . import data, metadata

__all__ = (
    'RTD_PROCESSORS',
    'RTD_META_CONSTRUCTORS'
)

RTD_PROCESSORS: Dict[str, FunctionType] = {
}
"""RTD table data processors."""

RTD_META_CONSTRUCTORS: Dict[str, FunctionType] = {
    'country': metadata.country
}
"""RTD metadata constructors for each table."""
