from datetime import timedelta

import pandas as pd

from dp_mobility_report import constants as const
from dp_mobility_report.model import utils
from dp_mobility_report.model.section import Section
from dp_mobility_report.privacy import diff_privacy


def get_dataset_statistics(mdreport, eps):
    if mdreport.evalu == True or eps is None:
        epsi = eps
    else:
        epsi = eps / 4

    # counts for complete and incomplete trips
    points_per_trip = diff_privacy.counts_dp(
        mdreport.df.reset_index().groupby(const.TID).count()["index"].value_counts(), 
        epsi, 
        2 * mdreport.max_trips_per_user, parallel=True
    )
    n_incomplete_trips = points_per_trip[1] if (1 in points_per_trip.index) else 0
    n_complete_trips = points_per_trip[2] if (2 in points_per_trip.index) else 0
    n_trips = n_incomplete_trips + n_complete_trips
    n_records = n_incomplete_trips + n_complete_trips * 2

    n_users = diff_privacy.counts_dp(mdreport.df[const.UID].nunique(), epsi, 1)
    n_locations = diff_privacy.counts_dp(
        mdreport.df.groupby([const.LAT, const.LNG]).ngroups,
        epsi,
        2 * mdreport.max_trips_per_user,
    )

    stats = dict(
        n_records=n_records,
        n_trips=n_trips,
        n_complete_trips=n_complete_trips,
        n_incomplete_trips=n_incomplete_trips,
        n_users=n_users,
        n_locations=n_locations,
    )
    return Section(
        data=stats,
        privacy_budget=eps
    )


def get_missing_values(mdreport, eps):
    columns = [const.UID, const.TID, const.DATETIME, const.LAT, const.LNG]

    if eps is not None:
        epsi = eps / len(columns)
    else:
        epsi = eps
    missings = dict((len(mdreport.df) - mdreport.df.count())[columns])

    for col in columns:
        missings[col] = diff_privacy.counts_dp(
            missings[col], epsi, 2 * mdreport.max_trips_per_user, nonzero=False
        )

    return Section(
        data=missings,
        privacy_budget=eps
    )


def get_trips_over_time(mdreport, eps):
    if mdreport.evalu == True or eps is None:
        epsi = eps
        epsi_quant = epsi
    else:
        epsi = eps / 6
        epsi_quant = 5 * epsi
    df_trip = mdreport.df[(mdreport.df.point_type == const.END)] # only count each trip once
    dp_quartiles = diff_privacy.quartiles_dp(
        df_trip.datetime, epsi_quant, mdreport.max_trips_per_user
    )

    # cut based on dp min and max values
    trips_over_time, _ = utils.cut_outliers( # don't disclose outliers to the as the boundaries are not defined through user input
        df_trip.datetime, min_value=dp_quartiles["min"], max_value=dp_quartiles["max"]
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
    trip_count.datetime = trip_count.datetime.dt.date
    trip_count.trip_count = diff_privacy.counts_dp(
        trip_count["trip_count"],
        epsi,
        mdreport.max_trips_per_user,
        parallel=True,
        nonzero=False,
    )

    return Section(
        data=trip_count,
        privacy_budget=eps,
        datetime_precision=datetime_precision,
        quartiles=dp_quartiles
    )


def get_trips_per_weekday(mdreport, eps):
    mdreport.df.loc[:, const.DATE] = mdreport.df.datetime.dt.date
    mdreport.df.loc[:, const.DAY_NAME] = mdreport.df.datetime.dt.day_name()

    trips_per_weekday = (
        mdreport.df[mdreport.df.point_type == const.END]  # count trips not records
        .groupby([const.DAY_NAME])
        .count()
        .tid
    )

    dp_trips_per_weekday = diff_privacy.counts_dp(
        trips_per_weekday,
        eps,
        mdreport.max_trips_per_user,
        parallel=True,
        nonzero=False,
    )

    return Section(
        data=dp_trips_per_weekday,
        privacy_budget=eps
        )



def get_trips_per_hour(mdreport, eps):
    hour_weekday = mdreport.df.groupby([const.HOUR, const.IS_WEEKEND, const.POINT_TYPE]).count().tid
    hour_weekday.name = "count"

    dp_hour_weekday = diff_privacy.counts_dp(
        hour_weekday, eps, mdreport.max_trips_per_user, parallel=True
    ).reset_index()

    dp_hour_weekday[const.TIME_CATEGORY] = (
        dp_hour_weekday[const.IS_WEEKEND] + "_" + dp_hour_weekday[const.POINT_TYPE]
    )
    dp_hour_weekday = dp_hour_weekday[[const.HOUR, const.TIME_CATEGORY, "count"]]

    return Section(
        data=dp_hour_weekday,
        privacy_budget=eps
        )
