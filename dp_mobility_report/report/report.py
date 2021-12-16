from tqdm.auto import tqdm


from dp_mobility_report.model import (
    overview,
    place_analysis,
    od_analysis,
    user_analysis,
)


def report_elements(mdreport, disable_progress_bar=False):

    report = dict()

    with tqdm(  # progress bar
        total=4, desc="Create report", disable=disable_progress_bar
    ) as pbar:
        if ("all" in mdreport.analysis_selection) | (
            "overview" in mdreport.analysis_selection
        ):
            report = {**report, **add_overview_elements(mdreport)}
        pbar.update()

        if ("all" in mdreport.analysis_selection) | (
            "place_analysis" in mdreport.analysis_selection
        ):
            report = {**report, **add_place_analysis_elements(mdreport)}
        pbar.update()

        if ("all" in mdreport.analysis_selection) | (
            "od_analysis" in mdreport.analysis_selection
        ):
            _od_shape = od_analysis.get_od_shape(mdreport.df, mdreport.tessellation)
            report = {**report, **add_od_analysis_elements(mdreport, _od_shape)}
        pbar.update()

        if ("all" in mdreport.analysis_selection) | (
            "user_analysis" in mdreport.analysis_selection
        ):
            report = {**report, **add_user_analysis_elements(mdreport)}
        pbar.update()

    return report


def add_overview_elements(mdreport):
    if mdreport.privacy_budget == None or mdreport.evalu == True:
        epsilon = mdreport.privacy_budget
    else:
        epsilon = mdreport.privacy_budget / 6
    return dict(
        ds_statistics=overview.get_dataset_statistics(mdreport, epsilon),
        # extra_var_counts=(
        #     None
        #     if mdreport.extra_var is None
        #     else overview.get_extra_var_counts(mdreport,epsilon)
        # ),
        missing_values=overview.get_missing_values(mdreport, epsilon),
        trips_over_time_section=overview.get_trips_over_time(mdreport, epsilon),
        trips_per_weekday=overview.get_trips_per_weekday(mdreport, epsilon),
        trips_per_hour=overview.get_trips_per_hour(mdreport, epsilon),
    )


def add_place_analysis_elements(mdreport):
    if mdreport.privacy_budget == None or mdreport.evalu == True:
        epsilon = mdreport.privacy_budget
    else:
        epsilon = mdreport.privacy_budget / 2
    counts_per_tile_section = place_analysis.get_counts_per_tile(mdreport, epsilon)
    return dict(
        counts_per_tile_section=counts_per_tile_section,
        counts_per_tile_timewindow=place_analysis.get_counts_per_tile_timewindow(
            mdreport, epsilon
        ),
    )


def add_od_analysis_elements(mdreport, _od_shape):
    if mdreport.privacy_budget == None or mdreport.evalu == True:
        epsilon = mdreport.privacy_budget
    else:
        epsilon = mdreport.privacy_budget / 3
    return dict(
        od_flows=od_analysis.get_od_flows(_od_shape, epsilon, mdreport),
        travel_time_section=od_analysis.get_travel_time(
            _od_shape,
            epsilon,
            mdreport.max_trips_per_user,
            mdreport.max_travel_time,
            mdreport.bin_size_travel_time,
        ),
        jump_length_section=od_analysis.get_jump_length(
            _od_shape,
            epsilon,
            mdreport.max_trips_per_user,
            mdreport.max_jump_length,
            mdreport.bin_size_jump_length,
        ),
    )


def add_user_analysis_elements(mdreport):
    if mdreport.privacy_budget == None or mdreport.evalu == True:
        epsilon = mdreport.privacy_budget
    else:
        epsilon = mdreport.privacy_budget / 7
    return dict(
        traj_per_user_section=user_analysis.get_traj_per_user(mdreport, epsilon),
        user_time_delta_section=user_analysis.get_user_time_delta(mdreport, epsilon),
        radius_gyration_section=user_analysis.get_radius_of_gyration(mdreport, epsilon),
        location_entropy_section=user_analysis.get_location_entropy(mdreport, epsilon),
        user_tile_count_section=user_analysis.get_user_tile_count(mdreport, epsilon),
        uncorrelated_entropy_section=user_analysis.get_uncorrelated_entropy(
            mdreport, epsilon
        ),
    )
