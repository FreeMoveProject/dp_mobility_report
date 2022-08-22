from datetime import timedelta
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from dp_mobility_report import DpMobilityReport

from dp_mobility_report import constants as const
from dp_mobility_report.model import m_utils
from dp_mobility_report.model.section import Section
from dp_mobility_report.privacy import diff_privacy


def get_dataset_statistics(mreport: "DpMobilityReport", eps: Optional[float]) -> Section:
    epsi = m_utils.get_epsi(mreport.evalu, eps, 4)

    # counts for complete and incomplete trips
    points_per_trip = (
        mreport.df.reset_index().groupby(const.TID).count()["index"].value_counts()
    )
    n_incomplete_trips = 0 if 1 not in points_per_trip else points_per_trip[1]
    n_incomplete_trips = diff_privacy.count_dp(
        n_incomplete_trips,
        epsi,
        mreport.max_trips_per_user,
    )
    moe_incomplete_trips = diff_privacy.laplace_margin_of_error(
        0.95, epsi, mreport.max_trips_per_user
    )
    ci95_incomplete_trips = diff_privacy.conf_interval(
        n_incomplete_trips, moe_incomplete_trips
    )

    n_complete_trips = 0 if 2 not in points_per_trip else points_per_trip[2]
    n_complete_trips = diff_privacy.count_dp(
        n_complete_trips,
        epsi,
        2 * mreport.max_trips_per_user,
    )
    moe_complete_trips = diff_privacy.laplace_margin_of_error(
        0.95, epsi, 2 * mreport.max_trips_per_user
    )
    ci95_complete_trips = diff_privacy.conf_interval(
        n_complete_trips, moe_complete_trips
    )

    n_trips = n_incomplete_trips + n_complete_trips
    if n_trips == 0:
        n_trips = None  # trips cannot be None
        moe_trips = (moe_incomplete_trips + moe_complete_trips) / 2
    else:
        moe_trips = (
            n_incomplete_trips * moe_incomplete_trips
            + n_complete_trips * moe_complete_trips
        ) / n_trips
    ci95_trips = diff_privacy.conf_interval(n_trips, moe_trips)

    n_records = None if n_trips is None else (n_incomplete_trips + n_complete_trips * 2)
    if n_records is None:
        moe_records = (moe_incomplete_trips + moe_complete_trips * 2) / 2
    else:
        moe_records = (
            n_incomplete_trips * moe_incomplete_trips
            + n_complete_trips * 2 * moe_complete_trips
        ) / n_records
    ci95_records = diff_privacy.conf_interval(n_records, moe_records)

    n_users = diff_privacy.count_dp(
        mreport.df[const.UID].nunique(), epsi, 1, nonzero=True
    )
    moe_users = diff_privacy.laplace_margin_of_error(0.95, epsi, 1)
    ci95_users = diff_privacy.conf_interval(n_users, moe_users)

    n_locations = diff_privacy.count_dp(
        mreport.df.groupby([const.LAT, const.LNG]).ngroups,
        epsi,
        2 * mreport.max_trips_per_user,
        nonzero=True,
    )
    moe_locations = diff_privacy.laplace_margin_of_error(
        0.95, epsi, 2 * mreport.max_trips_per_user
    )
    ci95_locations = diff_privacy.conf_interval(n_locations, moe_locations)

    stats = {
        "n_records": n_records,
        "n_trips": n_trips,
        "n_complete_trips": n_complete_trips,
        "n_incomplete_trips": n_incomplete_trips,
        "n_users": n_users,
        "n_locations": n_locations,
    }
    conf_interval = {
        "ci95_complete_trips": ci95_complete_trips,
        "ci95_incomplete_trips": ci95_incomplete_trips,
        "ci95_trips": ci95_trips,
        "ci95_records": ci95_records,
        "ci95_users": ci95_users,
        "ci95_locations": ci95_locations,
    }
    return Section(data=stats, privacy_budget=eps, conf_interval=conf_interval)


def get_missing_values(mreport: "DpMobilityReport", eps: Optional[float]) -> Section:
    columns = [const.UID, const.TID, const.DATETIME, const.LAT, const.LNG]
    epsi = m_utils.get_epsi(mreport.evalu, eps, len(columns))

    missings = dict((len(mreport.df) - mreport.df.count())[columns])

    moe = diff_privacy.laplace_margin_of_error(
        0.95, epsi, 2 * mreport.max_trips_per_user
    )
    conf_interval = {}
    for col in columns:
        missings[col] = diff_privacy.count_dp(
            missings[col], epsi, 2 * mreport.max_trips_per_user
        )
        conf_interval["ci95_" + col] = diff_privacy.conf_interval(missings[col], moe)

    return Section(data=missings, privacy_budget=eps, conf_interval=conf_interval)


def get_trips_over_time(mreport: "DpMobilityReport", eps: Optional[float]) -> Section:
    epsi = m_utils.get_epsi(mreport.evalu, eps, 3)
    epsi_limits = epsi * 2 if epsi is not None else None

    df_trip = mreport.df[
        (mreport.df[const.POINT_TYPE] == const.END)
    ]  # only count each trip once
    dp_bounds = diff_privacy.bounds_dp(
        df_trip[const.DATETIME], epsi_limits, mreport.max_trips_per_user
    )

    # cut based on dp min and max values
    (
        trips_over_time
    ) = m_utils.cut_outliers(  # don't disclose outliers to the as the boundaries are not defined through user input
        df_trip[const.DATETIME],
        min_value=dp_bounds[0],
        max_value=dp_bounds[1],
    )

    # only use date and remove time
    dp_bounds = pd.Series(dp_bounds).dt.date

    # different aggregations based on range of dates
    range_of_days = dp_bounds[1] - dp_bounds[0]
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
        trip_count.set_index(const.DATETIME)
        .resample(resample, label="left")
        .count()
        .reset_index()
    )
    trip_count[const.DATETIME] = trip_count[const.DATETIME].dt.date
    trip_count["trip_count"] = diff_privacy.counts_dp(
        trip_count["trip_count"].values,
        epsi,
        mreport.max_trips_per_user,
    )

    moe_laplace = diff_privacy.laplace_margin_of_error(
        0.95, epsi, mreport.max_trips_per_user
    )

    # as percent instead of absolute values
    trip_sum = np.sum(trip_count["trip_count"])
    if trip_sum != 0:
        trip_count["trip_count"] = trip_count["trip_count"] / trip_sum * 100
        moe_laplace = moe_laplace / trip_sum * 100

    quartiles = pd.Series({"min": dp_bounds[0], "max": dp_bounds[1]})

    return Section(
        data=trip_count,
        privacy_budget=eps,
        datetime_precision=datetime_precision,
        quartiles=quartiles,
        margin_of_error_laplace=moe_laplace,
    )


def get_trips_per_weekday(mreport: "DpMobilityReport", eps: Optional[float]) -> Section:
    mreport.df.loc[:, const.DATE] = mreport.df[const.DATETIME].dt.date
    mreport.df.loc[:, const.DAY_NAME] = mreport.df[const.DATETIME].dt.day_name()
    mreport.df.loc[:, const.WEEKDAY] = mreport.df[const.DATETIME].dt.weekday

    trips_per_weekday = (
        mreport.df[mreport.df[const.POINT_TYPE] == const.END]  # count trips not records
        .sort_values(const.WEEKDAY)
        .groupby([const.DAY_NAME], sort=False)
        .count()[const.TID]
    )

    trips_per_weekday = pd.Series(
        index=trips_per_weekday.index,
        data=diff_privacy.counts_dp(
            trips_per_weekday.values,
            eps,
            mreport.max_trips_per_user,
        ),
    )
    moe = diff_privacy.laplace_margin_of_error(0.95, eps, mreport.max_trips_per_user)

    trip_sum = np.sum(trips_per_weekday)
    if trip_sum != 0:
        trips_per_weekday = trips_per_weekday / trip_sum * 100
        moe = moe / trip_sum * 100

    return Section(
        data=trips_per_weekday, privacy_budget=eps, margin_of_error_laplace=moe
    )


def get_trips_per_hour(mreport: "DpMobilityReport", eps: Optional[float]) -> Section:
    hour_weekday = mreport.df.groupby(
        [const.HOUR, const.IS_WEEKEND, const.POINT_TYPE]
    ).count()[const.TID]
    hour_weekday.name = "count"

    hour_weekday = hour_weekday.reset_index()
    hour_weekday["count"] = diff_privacy.counts_dp(
        hour_weekday["count"], eps, mreport.max_trips_per_user
    )

    hour_weekday[const.TIME_CATEGORY] = (
        hour_weekday[const.IS_WEEKEND] + "_" + hour_weekday[const.POINT_TYPE]
    )
    moe = diff_privacy.laplace_margin_of_error(0.95, eps, mreport.max_trips_per_user)

    # as percent instead of absolute values
    trip_sum = np.sum(
        hour_weekday[hour_weekday.point_type == const.END]["count"]
    )  # only use ends to get sum of trips
    if trip_sum != 0:
        hour_weekday["count"] = hour_weekday["count"] / trip_sum * 100
        moe = moe / trip_sum * 100

    return Section(
        data=hour_weekday[[const.HOUR, const.TIME_CATEGORY, "count"]],
        privacy_budget=eps,
        margin_of_error_laplace=moe,
    )
