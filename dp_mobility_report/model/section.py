from dp_mobility_report import constants as const


class Section:
    """Class for single report sections. Attribute 'data' contains the content of the report element.
    'privacy_budget' contains information on the used privacy_budget to create this report element.
    'n_outliers' contains information of outliers, if there are potential outliers for this report element - otherwise it defaults to `None`.
    Potential further attributes: `quartiles`, `datetime_precision`.
    """

    def __init__(self, data, privacy_budget, n_outliers=None, **kwargs) -> None:

        self.data = data
        self.n_outliers = n_outliers
        self.privacy_budget = privacy_budget
        if const.QUARTILES in kwargs:
            self.quartiles = kwargs[const.QUARTILES]
        if const.DATETIME_PRECISION in kwargs:
            self.datetime_precision = kwargs[const.DATETIME_PRECISION]

    def __str__(self):
        print_output = {
            "data": self.data,
            "privacy_budget": self.privacy_budget,
            "n_outlier": self.n_outliers,
        }
        if hasattr(self, const.QUARTILES):
            print_output[const.QUARTILES] = self.quartiles
        if hasattr(self, const.DATETIME_PRECISION):
            print_output[const.DATETIME_PRECISION] = self.datetime_precision
        return print_output.__str__()
