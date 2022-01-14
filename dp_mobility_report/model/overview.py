from datetime import timedelta
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from dp_mobility_report.md_report import MobilityDataReport

from dp_mobility_report import constants as const
from dp_mobility_report.model import m_utils
from dp_mobility_report.model.section import Section
from dp_mobility_report.privacy import diff_privacy


def get_dataset_statistics(
    mdreport: "MobilityDataReport", eps: Optional[float]
) -> Section:
    if mdreport.evalu or eps is None:
        epsi = eps
    else:
        epsi = eps / 5

    # counts for complete and incomplete trips
    points_per_trip = (
        mdreport.df.reset_index().groupby(const.TID).count()["index"].value_counts()
    )
    n_incomplete_trips = 0 if 1 not in points_per_trip else points_per_trip[1]
    n_incomplete_trips = diff_privacy.count_dp(
        n_incomplete_trips,
        epsi,
        mdreport.max_trips_per_user,
    )
    n_complete_trips = 0 if 2 not in points_per_trip else points_per_trip[2]
    n_complete_trips = diff_privacy.count_dp(
        n_complete_trips,
        epsi,
        2 * mdreport.max_trips_per_user,
    )
    n_trips = n_incomplete_trips + n_complete_trips
    n_trips = None if n_trips == 0 else n_trips
    n_records = None if n_trips == 0 else (n_incomplete_trips + n_complete_trips * 2)

    n_users = diff_privacy.count_dp(
        mdreport.df[const.UID].nunique(), epsi, 1, nonzero=True
    )
    n_locations = diff_privacy.count_dp(
        mdreport.df.groupby([const.LAT, const.LNG]).ngroups,
        epsi,
        2 * mdreport.max_trips_per_user,
        nonzero=True,
    )

    stats = {
        "n_records": n_records,
        "n_trips": n_trips,
        "n_complete_trips": n_complete_trips,
        "n_incomplete_trips": n_incomplete_trips,
        "n_users": n_users,
        "n_locations": n_locations,
    }
    return Section(data=stats, privacy_budget=eps)


def get_missing_values(mdreport: "MobilityDataReport", eps: Optional[float]) -> Section:
    columns = [const.UID, const.TID, const.DATETIME, const.LAT, const.LNG]
    epsi = eps / len(columns) if eps is not None else None
    missings = dict((len(mdreport.df) - mdreport.df.count())[columns])

    for col in columns:
        missings[col] = diff_privacy.count_dp(
            missings[col], epsi, 2 * mdreport.max_trips_per_user
        )

    return Section(data=missings, privacy_budget=eps)


def get_trips_over_time(
    mdreport: "MobilityDataReport", eps: Optional[float]
) -> Section:
    if mdreport.evalu or eps is None:
        epsi = eps
        epsi_quant = epsi
    else:
        epsi = eps / 6
        epsi_quant = 5 * epsi
    df_trip = mdreport.df[
        (mdreport.df[const.POINT_TYPE] == const.END)
    ]  # only count each trip once
    dp_quartiles = diff_privacy.quartiles_dp(
        df_trip[const.DATETIME], epsi_quant, mdreport.max_trips_per_user
    )

    # cut based on dp min and max values
    (
        trips_over_time,
        _,
    ) = m_utils.cut_outliers(  # don't disclose outliers to the as the boundaries are not defined through user input
        df_trip[const.DATETIME],
        min_value=dp_quartiles["min"],
        max_value=dp_quartiles["max"],
    )

    # only use date and remove time
    dp_quartiles = dp_quartiles.dt.date

    # different aggregations based on range of dates
    range_of_days = dp_quartiles["max"] - dp_quartiles["min"]
    if range_of_days > timedelta(days=712):  # more than two years (102 weeks)
        resample = "M"
        datetime_precision = const.PREC_MONTH
    if range_of_days > timedelta(days=90):  # more than three months
        resample = "W-Mon"
        datetime_precision = const.PREC_WEEK
    else:
        resample = "1D"
        datetime_precision = const.PREC_DATE

    trip_count = pd.DataFrame(trips_over_time)
    trip_count.loc[:, "trip_count"] = 1
    trip_count = (
        trip_count.set_index(const.DATETIME).resample(resample).count().reset_index()
    )
    trip_count[const.DATETIME] = trip_count[const.DATETIME].dt.date
    trip_count["trip_count"] = diff_privacy.counts_dp(
        trip_count["trip_count"].values,
        epsi,
        mdreport.max_trips_per_user,
    )

    return Section(
        data=trip_count,
        privacy_budget=eps,
        datetime_precision=datetime_precision,
        quartiles=dp_quartiles,
    )


def get_trips_per_weekday(
    mdreport: "MobilityDataReport", eps: Optional[float]
) -> Section:
    mdreport.df.loc[:, const.DATE] = mdreport.df[const.DATETIME].dt.date
    mdreport.df.loc[:, const.DAY_NAME] = mdreport.df[const.DATETIME].dt.day_name()

    trips_per_weekday = (
        mdreport.df[
            mdreport.df[const.POINT_TYPE] == const.END
        ]  # count trips not records
        .groupby([const.DAY_NAME])
        .count()[const.TID]
    )

    trips_per_weekday = pd.Series(
        index=trips_per_weekday.index,
        data=diff_privacy.counts_dp(
            trips_per_weekday.values,
            eps,
            mdreport.max_trips_per_user,
        ),
    )

    return Section(data=trips_per_weekday, privacy_budget=eps)


def get_trips_per_hour(mdreport: "MobilityDataReport", eps: Optional[float]) -> Section:
    hour_weekday = mdreport.df.groupby(
        [const.HOUR, const.IS_WEEKEND, const.POINT_TYPE]
    ).count()[const.TID]
    hour_weekday.name = "count"

    hour_weekday = hour_weekday.reset_index()
    hour_weekday["count"] = diff_privacy.counts_dp(
        hour_weekday["count"], eps, mdreport.max_trips_per_user
    )

    hour_weekday[const.TIME_CATEGORY] = (
        hour_weekday[const.IS_WEEKEND] + "_" + hour_weekday[const.POINT_TYPE]
    )

    return Section(
        data=hour_weekday[[const.HOUR, const.TIME_CATEGORY, "count"]],
        privacy_budget=eps,
    )
