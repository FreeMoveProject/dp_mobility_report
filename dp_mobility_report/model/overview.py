from datetime import timedelta

import numpy as np
import pandas as pd

from dp_mobility_report.model import utils
from dp_mobility_report.model.section import Section
from dp_mobility_report.privacy import diff_privacy


def get_dataset_statistics(mdreport, eps):
    if mdreport.evalu == True or eps is None:
        epsi = eps
    else:
        epsi = eps / 4

    pointsTrip = mdreport.df.groupby("tid").count().uid.value_counts()
    n_start_end = diff_privacy.counts_dp(
        pointsTrip, epsi, 2 * mdreport.max_trips_per_user, parallel=True
    )

    ntrips_single = n_start_end[1] if (1 in n_start_end.index) else 0
    ntrips_double = n_start_end[2] if (2 in n_start_end.index) else 0
    n_trips = ntrips_single + ntrips_double
    n_records = ntrips_single + ntrips_double * 2

    n_users = diff_privacy.counts_dp(mdreport.df.uid.nunique(), epsi, 1)
    n_places = diff_privacy.counts_dp(
        mdreport.df.groupby(["lat", "lng"]).ngroups,
        epsi,
        2 * mdreport.max_trips_per_user,
    )

    stats = dict(
        n_records=n_records,
        n_trips=n_trips,
        ntrips_double=ntrips_double,
        ntrips_single=ntrips_single,
        n_users=n_users,
        n_places=n_places,
    )

    # if mdreport.extra_var is not None:
    #     # todo saskia consistent with get_extra_var_counts
    #     n_extra_var = diff_privacy.counts_dp(mdreport.df[mdreport.extra_var].nunique(),epsi,2*mdreport.max_trips_per_user,parallel=True)
    #     stats["n_extra_var"] = n_extra_var[0]

    return stats


# TODO: problem - "leaking of groupby keys"
# def get_extra_var_counts(mdreport,eps):
#     extra_var_value_counts = diff_privacy.counts_dp(mdreport.df[mdreport.extra_var].value_counts(),eps,2*mdreport.max_trips_per_user,parallel=True)
#     #if eps==None:
#      #   extra_var_value_counts = df[extra_var].value_counts()

#     extra_var_perc = pd.Series(
#         round(extra_var_value_counts / sum(extra_var_value_counts) * 100),
#         name="perc",
#     )
#     return pd.concat([extra_var_value_counts, extra_var_perc], axis=1)


def get_missing_values(mdreport, eps):

    # if mdreport.extra_var is not None:
    #     columns = ["uid", "tid", "datetime", "lat", "lng", mdreport.extra_var]
    # else:
    columns = ["uid", "tid", "datetime", "lat", "lng"]

    if eps is not None:
        epsi = eps / len(columns)
    else:
        epsi = eps
    missings = dict((len(mdreport.df) - mdreport.df.count())[columns])

    for col in columns:
        missings[col] = diff_privacy.counts_dp(
            missings[col], epsi, 2 * mdreport.max_trips_per_user, nonzero=False
        )

    return missings


def get_trips_over_time(mdreport, eps):
    if mdreport.evalu == True or eps is None:
        epsi = eps
        epsi_quant = epsi
    else:
        epsi = eps / 6
        epsi_quant = 5 * epsi
    df_trip = mdreport.df[(mdreport.df.point_type == "end")]
    dp_quartiles = diff_privacy.quartiles_dp(
        df_trip.datetime, epsi_quant, mdreport.max_trips_per_user
    )

    # cut based on dp min and max values
    trips_over_time, _ = utils.cut_outliers(
        df_trip.datetime, min_value=dp_quartiles["min"], max_value=dp_quartiles["max"]
    )

    # only keep date of datetime
    dp_quartiles = dp_quartiles.dt.date

    # different aggregations based on range of dates
    range_of_days = dp_quartiles["max"] - dp_quartiles["min"]
    if range_of_days > timedelta(days=712):  # more than two years (102 weeks)
        resample = "M"
        date_aggregation_level = "month"
    if range_of_days > timedelta(days=90):  # more than three months
        resample = "W-Mon"
        date_aggregation_level = "week"
    else:
        resample = "1D"
        date_aggregation_level = "date"

    trip_count = pd.DataFrame(trips_over_time)
    trip_count.loc[:, "trip_count"] = 1
    trip_count = (
        trip_count.set_index("datetime").resample(resample).count().reset_index()
    )
    trip_count.datetime = trip_count.datetime.dt.date
    # real_entropy_section=user_analysis.get_real_entropy(_tdf,epsilon, mdreport.evalu)
    trip_count.trip_count = diff_privacy.counts_dp(
        trip_count["trip_count"],
        epsi,
        mdreport.max_trips_per_user,
        parallel=True,
        nonzero=False,
    )

    return Section(
        data=trip_count,
        n_outliers=None,
        date_aggregation_level=date_aggregation_level,
        quartiles=dp_quartiles,
    )


def get_trips_per_weekday(mdreport, eps):

    mdreport.df.loc[:, "date"] = mdreport.df.datetime.dt.date
    mdreport.df.loc[:, "day_name"] = mdreport.df.datetime.dt.day_name()

    trips_per_weekday = (
        mdreport.df[mdreport.df.point_type == "end"]  # count trips not records
        .groupby(["day_name"])
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

    return dp_trips_per_weekday


def get_trips_per_hour(mdreport, eps):
    hour_weekday = mdreport.df.groupby(["hour", "is_weekend", "point_type"]).count().tid
    hour_weekday.name = "count"

    dp_hour_weekday = diff_privacy.counts_dp(
        hour_weekday, eps, mdreport.max_trips_per_user, parallel=True
    ).reset_index()

    dp_hour_weekday["time_category"] = (
        dp_hour_weekday.is_weekend + "_" + dp_hour_weekday.point_type
    )
    dp_hour_weekday = dp_hour_weekday[["hour", "time_category", "count"]]

    return dp_hour_weekday
