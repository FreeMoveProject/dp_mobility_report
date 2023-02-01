import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

from dp_mobility_report import constants as const
from dp_mobility_report.benchmark.b_utils import default_measure_selection
from dp_mobility_report.model.section import DfSection


# only select analyses that are present in both reports
def combine_analysis_exclusion(
    analysis_exclusion_alternative: List[str], analysis_exclusion_base: List[str]
) -> List[str]:
    analysis_exclusion = list(
        set(analysis_exclusion_alternative + analysis_exclusion_base)
    )
    analysis_exclusion.sort(key=lambda i: const.ELEMENTS.index(i))
    return analysis_exclusion


def _resample_to_prec(
    df: pd.DataFrame, current_precision: str, target_precision: str
) -> pd.DataFrame:
    if current_precision == target_precision:
        return df
    precision_names = {
        const.PREC_MONTH: "M",
        const.PREC_WEEK: "W-MON",
        const.PREC_DATE: "1D",
    }
    df[const.DATETIME] = pd.to_datetime(df[const.DATETIME])
    df_resampled = (
        df.set_index(const.DATETIME)
        .resample(precision_names[target_precision], label="left")
        .sum()
        .reset_index()
    )
    df_resampled[const.DATETIME] = df_resampled[const.DATETIME].dt.date
    return df_resampled


def unify_trips_over_time(base: DfSection, alternative: DfSection) -> None:
    if base.datetime_precision != alternative.datetime_precision:
        if const.PREC_MONTH in [
            base.datetime_precision,
            alternative.datetime_precision,
        ]:
            base.data = _resample_to_prec(
                base.data, base.datetime_precision, const.PREC_MONTH
            )
            base.datetime_precision = const.PREC_MONTH
            alternative.data = _resample_to_prec(
                alternative.data, alternative.datetime_precision, const.PREC_MONTH
            )
            alternative.datetime_precision = const.PREC_MONTH
        elif const.PREC_WEEK in [
            base.datetime_precision,
            alternative.datetime_precision,
        ]:
            base.data = _resample_to_prec(
                base.data, base.datetime_precision, const.PREC_WEEK
            )
            base.datetime_precision = const.PREC_WEEK
            alternative.data = _resample_to_prec(
                alternative.data, alternative.datetime_precision, const.PREC_WEEK
            )
            alternative.datetime_precision = const.PREC_WEEK

    missing_dates_base = list(
        set(alternative.data[const.DATETIME]) - set(base.data[const.DATETIME])
    )
    missing_dates_alternative = list(
        set(base.data[const.DATETIME]) - set(alternative.data[const.DATETIME])
    )
    base.data = pd.concat(
        [
            base.data,
            pd.DataFrame(
                data={const.DATETIME: missing_dates_base, "trip_count": 0, "trips": 0}
            ),
        ]
    ).sort_values(const.DATETIME)
    alternative.data = pd.concat(
        [
            alternative.data,
            pd.DataFrame(
                data={
                    const.DATETIME: missing_dates_alternative,
                    "trip_count": 0,
                    "trips": 0,
                }
            ),
        ]
    ).sort_values(const.DATETIME)


def _pad_missing_values(
    hist_bins: np.array, hist_values: np.array, bins_union: np.array
) -> np.array:
    missing_bins = [i for i in bins_union if i not in hist_bins]
    n_missing_left = sum(missing_bins < min(hist_bins))
    n_missing_right = sum(missing_bins > max(hist_bins))
    return np.append(
        np.zeros(n_missing_left), np.append(hist_values, np.zeros(n_missing_right))
    )


# check if histograms of both reports have similar bins
# if bins are missing they can only be at the lower end of the bins as bin_range and max_bins are similar
def unify_histogram_bins(
    report_base: dict, report_alternative: dict, analysis_exclusion: list
) -> Tuple[dict, dict]:
    histograms = [
        const.TRAVEL_TIME,
        const.JUMP_LENGTH,
        const.RADIUS_OF_GYRATION,
        const.USER_TIME_DELTA,
        const.USER_TILE_COUNT,
        const.TRIPS_OVER_TIME,
        # const.TRIPS_PER_USER, TODO: how to handle?
    ]
    # const.MOBILITY_ENTROPY should already be similar: 0-1 with 0.1 bin range

    for hist in histograms:
        if hist not in analysis_exclusion:
            if hist == const.TRIPS_OVER_TIME:
                unify_trips_over_time(report_base[hist], report_alternative[hist])

            else:
                hist_base_values, hist_base_bins = report_base[hist].data
                hist_alternative_values, hist_alternative_bins = report_alternative[
                    hist
                ].data

                bins_union = np.union1d(hist_alternative_bins, hist_base_bins)

                hist_base_values = _pad_missing_values(
                    hist_base_bins, hist_base_values, bins_union
                )
                hist_alternative_values = _pad_missing_values(
                    hist_alternative_bins, hist_alternative_values, bins_union
                )

                report_base[hist].data = (hist_base_values, bins_union)
                report_alternative[hist].data = (hist_alternative_values, bins_union)
    return report_base, report_alternative


def validate_measure_selection(
    measure_selection: dict, analysis_exclusion: list
) -> dict:

    default_selection = default_measure_selection()

    if not isinstance(measure_selection, dict):
        raise TypeError("'measure selection' is not a dictionary")

    for key in list(measure_selection.keys()):
        if key not in default_selection.keys():
            warnings.warn(
                f"{key} is not a valid key (analysis) and will be removed from the measure selection."
            )
            del measure_selection[key]

    for key, value in list(measure_selection.items()):

        if value not in const.SIMILARITY_MEASURES:
            warnings.warn(
                f"{value} is not a valid value (similarity measure) and will be removed from the measure selection."
            )
            del measure_selection[key]

    measure_selection = {**default_selection, **measure_selection}

    for analysis in analysis_exclusion:
        all_analysis = [
            item for item in measure_selection.keys() if item.startswith(analysis)
        ]
        for item in all_analysis:
            del measure_selection[item]

    return measure_selection


def validate_top_n_ranking(top_n_ranking: List[int]) -> List[int]:
    if not isinstance(top_n_ranking, list):
        raise TypeError(
            f"Input parameter top_n_ranking is not a list. Instead: {top_n_ranking}."
        )
    for x in top_n_ranking:
        if not isinstance(x, int):
            raise TypeError(
                f"Input parameter top_n_ranking doest not consist of ints. Instead: {top_n_ranking}."
            )
        if x <= 0:
            raise TypeError(
                f"Input parameter top_n_ranking doest not consist of ints greater 0. Instead: {top_n_ranking}."
            )
    return top_n_ranking
