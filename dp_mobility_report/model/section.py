from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class Section:
    """Class for single report sections. Attribute 'data' contains the content of the report element.
    'privacy_budget' contains information on the used privacy_budget to create this report element.
    'n_outliers' contains information of outliers, if there are potential outliers for this report element - otherwise it defaults to `None`.
    Potential further attributes: `quartiles`, `datetime_precision`.
    """

    data: Optional[Union[Tuple, dict, pd.Series, pd.DataFrame]]
    privacy_budget: Optional[float]
    n_outliers: Optional[int] = None
    quartiles: Optional[np.ndarray] = None
    datetime_precision: Optional[str] = None
