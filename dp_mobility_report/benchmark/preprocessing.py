from typing import List, Tuple

from dp_mobility_report import constants as const
import numpy as np

# only select analyses that are present in both reports
def combine_analysis_exclusion(
    analysis_exclusion_proposal: List[str], analysis_exclusion_benchmark: List[str]
) -> List[str]:
    analysis_exclusion = list(
        set(analysis_exclusion_proposal + analysis_exclusion_benchmark)
    )
    analysis_exclusion.sort(key=lambda i: const.ELEMENTS.index(i))
    return analysis_exclusion

# check if histograms of both reports have similar bins
# if bins are missing they can only be at the lower end of the bins as bin_range and max_bins are similar
def unify_histogram_bins(report_benchmark: dict, report_proposal: dict) -> Tuple[dict, dict]:
    histograms = [const.TRAVEL_TIME, const.JUMP_LENGTH, const.RADIUS_OF_GYRATION] # TODO: how to handle these? const.TRIPS_PER_USER, const.USER_TILE_COUNT, const.MOBILITY_ENTROPY

    for hist in histograms:
        hist_benchmark_values, hist_benchmark_bins = report_benchmark[hist].data
        hist_proposal_values, hist_proposal_bins = report_proposal[hist].data
        bins_union = np.union1d(hist_proposal_bins, hist_benchmark_bins)

        missing_in_benchmark = [i for i in bins_union if i not in hist_benchmark_bins]
        added_value_indizes = np.searchsorted(bins_union, missing_in_benchmark)
        for i in added_value_indizes:
            hist_benchmark_values = np.insert(hist_benchmark_values, i, 0)

        missing_in_proposal = [i for i in bins_union if i not in hist_proposal_bins]
        added_value_indizes = np.searchsorted(bins_union, missing_in_proposal)
        for i in added_value_indizes:
            hist_proposal_values = np.insert(hist_proposal_values, i, 0)

        report_benchmark[hist].data = (hist_benchmark_values, bins_union)
        report_proposal[hist].data = (hist_proposal_values, bins_union)
    return report_benchmark, report_proposal


# make sure there is a valid measure selected for each analysis
def validate_measure_selection(measure_selection):
    pass
