from dp_mobility_report import constants as const
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Union, Tuple

@dataclass
class Section:
    """Class for single report sections. Attribute 'data' contains the content of the report element.
    'privacy_budget' contains information on the used privacy_budget to create this report element.
    'n_outliers' contains information of outliers, if there are potential outliers for this report element - otherwise it defaults to `None`.
    Potential further attributes: `quartiles`, `datetime_precision`.
    """
    data: Union[pd.Series, np.array, Tuple]
    privacy_budget: float
    n_outliers: int = None
    quartiles: np.array = None
    datetime_precision: str = None

    # def __str__(self):
    #     print_output = {
    #         "data": self.data,
    #         "privacy_budget": self.privacy_budget,
    #         "n_outlier": self.n_outliers,
    #     }
    #     if hasattr(self, const.QUARTILES):
    #         print_output[const.QUARTILES] = self.quartiles
    #     if hasattr(self, const.DATETIME_PRECISION):
    #         print_output[const.DATETIME_PRECISION] = self.datetime_precision
    #     return print_output.__str__()
