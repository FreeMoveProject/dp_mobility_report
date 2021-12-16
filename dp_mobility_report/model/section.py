class Section:
    """[summary]
    """

    def __init__(self, data, n_outliers, privacy_budget=None, **kwargs) -> None:

        self.data = data
        self.n_outliers = n_outliers
        self.privacy_budget = privacy_budget
        if "quartiles" in kwargs:
            self.quartiles = kwargs["quartiles"]
        if "date_aggregation_level" in kwargs:
            self.date_aggregation_level = kwargs["date_aggregation_level"]

    def __str__(self):
        print_output = dict(data=self.data, n_outlier=self.n_outliers)
        if hasattr(self, "quartiles"):
            print_output["quartiles"] = self.quartiles
        if hasattr(self, "data_aggregation_level"):
            print_output["date_aggregation_level"] = self.date_aggregation_level
        return print_output.__str__()
