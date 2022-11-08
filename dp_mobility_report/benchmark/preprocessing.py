from typing import List, Tuple

from dp_mobility_report import constants as const
import numpy as np

# only select analyses that are present in both reports
def combine_analysis_exclusion(
    analysis_exclusion_alternative: List[str], analysis_exclusion_base: List[str]
) -> List[str]:
    analysis_exclusion = list(
        set(analysis_exclusion_alternative + analysis_exclusion_base)
    )
    analysis_exclusion.sort(key=lambda i: const.ELEMENTS.index(i))
    return analysis_exclusion

# check if histograms of both reports have similar bins
# if bins are missing they can only be at the lower end of the bins as bin_range and max_bins are similar
def unify_histogram_bins(report_base: dict, report_alternative: dict) -> Tuple[dict, dict]:
    histograms = [const.TRAVEL_TIME, const.JUMP_LENGTH, const.RADIUS_OF_GYRATION] # TODO: how to handle these? const.TRIPS_PER_USER, const.USER_TILE_COUNT, const.MOBILITY_ENTROPY

    for hist in histograms:
        hist_base_values, hist_base_bins = report_base[hist].data
        hist_alternative_values, hist_alternative_bins = report_alternative[hist].data
        bins_union = np.union1d(hist_alternative_bins, hist_base_bins)

        missing_in_base = [i for i in bins_union if i not in hist_base_bins]
        added_value_indizes = np.searchsorted(bins_union, missing_in_base)
        for i in added_value_indizes:
            hist_base_values = np.insert(hist_base_values, i, 0)

        missing_in_alternative = [i for i in bins_union if i not in hist_alternative_bins]
        added_value_indizes = np.searchsorted(bins_union, missing_in_alternative)
        for i in added_value_indizes:
            hist_alternative_values = np.insert(hist_alternative_values, i, 0)

        report_base[hist].data = (hist_base_values, bins_union)
        report_alternative[hist].data = (hist_alternative_values, bins_union)
    return report_base, report_alternative


# make sure there is a valid measure selected for each analysis
def validate_measure_selection(measure_selection):
    pass
