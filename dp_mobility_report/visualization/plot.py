import math
from functools import partial
from typing import Callable, Optional, Tuple, Type, Union

import folium
import geopandas as gpd
import matplotlib
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from geojson import LineString
from geopandas import GeoDataFrame
from matplotlib import pyplot as plt
from pandas import DataFrame

import dp_mobility_report.constants as const
from dp_mobility_report.report.html.html_utils import get_centroids

sns.set_theme()


def format(value: Union[float, int], type: Type, ndigits: int = 2) -> Union[float, int]:
    value = type(value)
    if type == float:
        value = round(value, ndigits)
    return value


def histogram(
    hist: Tuple,
    x_axis_label: str,
    hist_alternative: Optional[Tuple] = None,
    min_value: Union[int, float] = None,
    y_axis_label: str = "Frequency",
    margin_of_error: Union[float, list] = None,
    margin_of_error_alternative: Union[float, list] = None,
    rotate_label: bool = False,
    x_axis_type: Type = float,
    ndigits_x_label: int = 2,
    figsize: tuple = (6.4, 4.8),
) -> mpl.figure.Figure:
    bins = hist[1]
    counts = hist[0]
    counts_alternative = hist_alternative[0] if hist_alternative else None

    # single integers (instead of bin ranges) as x axis labels
    if len(bins) == len(counts):
        labels = np.array([format(bin, x_axis_type) for bin in bins[:-1]])
        if bins[-1] == np.Inf:
            labels = np.append(labels, f"> {labels[-1]}")
        else:
            labels = np.append(labels, format(bins[-1], x_axis_type))
        # needed for margin of error (see below)
        upper_limits = bins

    # bin ranges as labels
    else:
        lower_limits = bins[:-1]
        upper_limits = bins[1:]

        labels = np.array(
            [
                f"[{format(x1, x_axis_type, ndigits_x_label)}\n - \n{format(x2, x_axis_type, ndigits_x_label)})"
                if x2 != np.Inf
                else f"≥ {format(x1, x_axis_type)}"
                for x1, x2 in zip(lower_limits, upper_limits)
            ]
        )

        if upper_limits[-1] != np.Inf:
            labels[-1] = (
                labels[-1][:-1] + "]"
            )  # fix label string of last bin to include last value

    # margin of error only for bins above min value
    if min_value:
        margin_of_error = [
            0 if (min_value > upper_limits[i]) else margin_of_error
            for i in range(0, len(upper_limits))
        ]

    return barchart(
        x=labels,
        y=counts,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        y_alternative=counts_alternative,
        margin_of_error=margin_of_error,
        margin_of_error_alternative=margin_of_error_alternative,
        rotate_label=rotate_label,
        figsize=figsize,
    )


def barchart(
    x: np.ndarray,
    y: np.ndarray,
    x_axis_label: str,
    y_axis_label: str,
    y_alternative: Optional[np.ndarray] = None,
    margin_of_error: Optional[Union[float, list]] = None,
    margin_of_error_alternative: Optional[Union[float, list]] = None,
    rotate_label: bool = False,
    figsize: tuple = (6.4, 4.8),
) -> mpl.figure.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    bar_base = ax.bar(
        x, y, yerr=margin_of_error, align="center", alpha=0.5, capsize=10, label="base"
    )
    if y_alternative is not None:
        bar_alt = ax.bar(
            x,
            y_alternative,
            yerr=margin_of_error_alternative,
            width=0.3,
            align="center",
            alpha=0.5,
            capsize=10,
            color=const.LIGHT_ORANGE,
            ecolor=const.LIGHT_ORANGE,
            label="alternative",
        )
        ax.legend(handles=[bar_base, bar_alt])
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel(x_axis_label)
    plt.xticks(x)
    plt.ylim(bottom=0)

    if rotate_label:
        plt.xticks(rotation=90)
    return fig


def linechart(
    data: DataFrame,
    x: str,
    y: str,
    x_axis_label: str,
    y_axis_label: str,
    x_alternative: Optional[str] = None,
    margin_of_error: float = None,
    add_diagonal: bool = False,
    rotate_label: bool = False,
    figsize: tuple = (6.4, 4.8),
) -> mpl.figure.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    if margin_of_error is not None:
        ax.fill_between(
            data[x],
            (data[y] - margin_of_error),
            (data[y] + margin_of_error),
            color="blue",
            alpha=0.1,
        )
    ax.plot(data[x], data[y])
    if x_alternative is not None:
        ax.plot(data[x_alternative], data[y], color=const.LIGHT_ORANGE)
        ax.legend({"base", "alt"})
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel(x_axis_label)
    ax.set_ylim(bottom=0)
    if add_diagonal:
        ax.plot([0, data[x].max()], [0, data[y].max()], const.GREY)

    if rotate_label:
        plt.xticks(rotation=90)

    return fig


def linechart_new(
    data: DataFrame,
    x: str,
    y: str,
    x_axis_label: str,
    y_axis_label: str,
    data_alternative: Optional[DataFrame] = None,
    margin_of_error: float = None,
    margin_of_error_alternative: float = None,
    add_diagonal: bool = False,
    rotate_label: bool = False,
    figsize: tuple = (6.4, 4.8),
) -> mpl.figure.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    if margin_of_error is not None:
        ax.fill_between(
            data[x],
            (data[y] - margin_of_error),
            (data[y] + margin_of_error),
            color=const.LIGHT_BLUE,
            alpha=0.1,
        )
    (line_base,) = ax.plot(data[x], data[y], color=const.LIGHT_BLUE, label="base")
    if data_alternative is not None:
        (line_alt,) = ax.plot(
            data_alternative[x],
            data_alternative[y],
            color=const.LIGHT_ORANGE,
            label="alternative",
        )
        ax.legend(handles=[line_base, line_alt])
        if margin_of_error_alternative is not None:
            ax.fill_between(
                data_alternative[x],
                (data_alternative[y] - margin_of_error_alternative),
                (data_alternative[y] + margin_of_error_alternative),
                color=const.LIGHT_ORANGE,
                alpha=0.1,
            )
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel(x_axis_label)
    ax.set_ylim(bottom=0)
    if add_diagonal:
        ax.plot([0, data[x].max()], [0, data[y].max()], const.GREY, alpha=0.4)

    if rotate_label:
        plt.xticks(rotation=90)

    return fig


def multi_linechart(
    data: DataFrame,
    x: str,
    y: str,
    color: str,
    x_axis_label: str,
    y_axis_label: str,
    style: Optional[str] = None,
    hue_order: Optional[list] = None,
    margin_of_error: Optional[float] = None,
    figsize: tuple = (9, 6),
) -> mpl.figure.Figure:
    fig = plt.figure(figsize=figsize)
    plot = fig.add_subplot(111)
    palette = ["#99065a", "#e289ba", "#2c6a19", "#99cd60"]

    sns.lineplot(
        data=data,
        x=x,
        y=y,
        hue=color,
        ax=plot,
        style=style,
        hue_order=hue_order,
        palette=palette,
    )
    plot.set_ylim(bottom=0)
    plot.set_ylabel(y_axis_label)
    plot.set_xlabel(x_axis_label)

    if margin_of_error is not None:
        for i in range(0, len(hue_order)):
            hue_data = data[data[color] == hue_order[i]]
            plot.fill_between(
                hue_data[x],
                (hue_data[y] - margin_of_error),
                (hue_data[y] + margin_of_error),
                color=palette[i],
                alpha=0.1,
            )
    return fig


def choropleth_map(
    counts_per_tile_gdf: GeoDataFrame,
    fill_color_name: str,
    scale_title: str = "Visit count",
    map: Optional[folium.Map] = None,
    is_cmap_diverging: bool = False,
    cmap: Optional[str] = None,
    min_scale: Optional[Union[int, float]] = None,
    max_scale: Optional[Union[int, float]] = None,
    aliases: Optional[list] = None,
    layer_name: str = "Visits",
    show: bool = True,
) -> folium.Map:
    poly_json = counts_per_tile_gdf.to_json()

    if not map:
        center_x, center_y = _get_center(counts_per_tile_gdf)
        map = _basemap(center_x, center_y)

    min_scale = (
        min_scale
        if min_scale is not None
        else counts_per_tile_gdf[fill_color_name].min()
    )

    max_scale = (
        max_scale
        if max_scale is not None
        else counts_per_tile_gdf[fill_color_name].max()
    )

    # color
    if not cmap:
        cmap = const.DIVERGING_CMAP if is_cmap_diverging else const.STANDARD_CMAP
    mpl_cmap = mpl.colormaps[cmap]

    if is_cmap_diverging:
        norm = mpl.colors.TwoSlopeNorm(vmin=min_scale, vcenter=0, vmax=max_scale)
    else:
        norm = mpl.colors.Normalize(vmin=min_scale, vmax=max_scale)

    def _hex_color(x: Union[float, int], mpl_cmap: mpl.colors.Colormap) -> str:
        rgb = norm(x)
        return matplotlib.colors.rgb2hex(mpl_cmap(rgb))

    def _get_color(x: dict, fill_color_name: str, mpl_cmap: mpl.colors.Colormap) -> str:
        if x["properties"][fill_color_name] is None:
            return "#8c8c8c"
        return _hex_color(x["properties"][fill_color_name], mpl_cmap)

    def _style_function(
        x: dict, fill_color_name: str, mpl_cmap: mpl.colors.Colormap
    ) -> dict:
        return {
            "fillColor": _get_color(x, fill_color_name, mpl_cmap),
            "color": const.GREY,
            "weight": 1.5,
            "fillOpacity": 0.6,
        }

    _style_function_partial = partial(
        _style_function, fill_color_name=fill_color_name, mpl_cmap=mpl_cmap
    )

    if "tile_name" in counts_per_tile_gdf:
        fields = ["tile_id", "tile_name", fill_color_name]
    else:
        fields = ["tile_id", fill_color_name]
    folium.GeoJson(
        poly_json,
        name=layer_name,
        overlay=True,
        show=show,
        style_function=_style_function_partial,
        popup=folium.GeoJsonPopup(fields=fields, aliases=aliases),
    ).add_to(map)

    # colorbar object to create custom legend
    colorbar, ax = plt.subplots(figsize=(6, 1))
    colorbar.subplots_adjust(bottom=0.5)
    ax.grid(False)
    colorbar.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="horizontal",
        label=scale_title,
    )
    # TODO: add legend directly to map

    return map, colorbar


def multi_choropleth_map(
    counts_per_tile_timewindow: DataFrame,
    tessellation: GeoDataFrame,
    is_cmap_diverging: bool = False,
    min_scale: Optional[Union[int, float]] = None,
    max_scale: Optional[Union[int, float]] = None,
    vcenter: Optional[Union[int, float]] = 0,
) -> mpl.figure.Figure:
    counts_per_tile_timewindow = tessellation[["tile_id", "geometry"]].merge(
        counts_per_tile_timewindow, left_on="tile_id", right_index=True, how="left"
    )

    # col1: tile_id, col2: geometry
    plot_count = counts_per_tile_timewindow.shape[1] - 2
    plots_per_row = 3
    row_count = math.ceil(plot_count / plots_per_row)
    fig, axes = plt.subplots(row_count, plots_per_row, figsize=(18, 12))

    # upper and lower bound
    if min_scale is None:
        min_scale = counts_per_tile_timewindow.iloc[:, 2:].min().min()
        min_scale = min_scale if not math.isnan(min_scale) else 0
    if max_scale is None:
        max_scale = counts_per_tile_timewindow.iloc[:, 2:].max().max()
        max_scale = max_scale if not math.isnan(max_scale) else 2

    # color
    if is_cmap_diverging:
        cmap = const.DIVERGING_CMAP
        min_scale = (
            vcenter - 1 if min_scale >= vcenter else min_scale
        )  # if all values are the same, there are no deviations from average, but matplotlib needs ascending order of minscale, vcenter, maxscale
        max_scale = vcenter + 1 if max_scale <= vcenter else max_scale
        norm = mpl.colors.TwoSlopeNorm(vmin=min_scale, vcenter=vcenter, vmax=max_scale)
    else:
        cmap = const.BASE_CMAP  # STANDARD_CMAP
        norm = mpl.colors.Normalize(vmin=min_scale, vmax=max_scale)

    for i in range(0, plots_per_row * row_count):
        facet_row = math.ceil((i - plots_per_row + 1) / plots_per_row)
        if row_count == 1:
            ax = axes[i % plots_per_row]
        else:
            ax = axes[facet_row][i % plots_per_row]
        ax.axis("off")

        if i < (
            counts_per_tile_timewindow.shape[1] - 2
        ):  # there might be more subplots than data - skip in that case
            column_name = counts_per_tile_timewindow.columns[i + 2]
            counts_per_tile_timewindow.iloc[:, [i + 2, 1]].plot(
                column=column_name,
                cmap=cmap,
                norm=norm,
                linewidth=0.1,
                ax=ax,
                edgecolor="#FFFFFF",
                missing_kwds={
                    "color": const.LIGHT_GREY,
                },
            )
            ax.set_title(column_name)

    # Create colorbar as a legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []  # add the colorbar to the figure
    # set the range for the choropleth
    plt.rcParams["axes.grid"] = False  # silence matplotlib warning
    fig.colorbar(sm, ax=axes)
    plt.rcParams["axes.grid"] = True

    return fig


def _get_center(gdf: GeoDataFrame) -> Tuple:
    center_x = (gdf.total_bounds[0] + gdf.total_bounds[2]) / 2
    center_y = (gdf.total_bounds[1] + gdf.total_bounds[3]) / 2
    return (center_x, center_y)


def _basemap(center_x: float, center_y: float, zoom_start: int = 10) -> folium.Map:
    return folium.Map(
        tiles="CartoDB positron",
        attr='&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>, &copy; <a href="https://carto.com/attributions">CARTO</a>',
        zoom_start=zoom_start,
        location=[center_y, center_x],
        control_scale=True,
    )


def ranking(
    x: Union[np.ndarray, pd.Series],
    x_axis_label: str,
    x_alternative: Union[np.ndarray, pd.Series] = None,
    y_labels: list = None,
    margin_of_error: Optional[float] = None,
    margin_of_error_alternative: Optional[float] = None,
    figsize: tuple = (7, 5),
) -> mpl.figure.Figure:
    y = list(range(1, len(x) + 1))[::-1]
    y_labels = y if y_labels is None else y_labels
    fig, ax = plt.subplots(figsize=figsize)

    if all(np.isnan(x)):
        x = np.zeros(len(x))
    bar_base = ax.errorbar(
        x,
        y,
        xerr=margin_of_error,
        fmt="o",
        ecolor="lightblue",
        elinewidth=5,
        capsize=10,
        label="base",
    )
    if x_alternative is not None:
        if all(np.isnan(x_alternative)):
            x_alternative = np.zeros(len(x_alternative))
        bar_alt = ax.errorbar(
            x_alternative,
            y,
            xerr=margin_of_error_alternative,
            fmt="o",
            ecolor=const.LIGHT_ORANGE,
            elinewidth=5,
            capsize=10,
            label="alternative",
        )
        ax.legend(handles=[bar_base, bar_alt])
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel("Rank")
    ax.set_xlabel(x_axis_label)
    return fig


def flows(
    basemap: folium.Map,
    data: pd.DataFrame,
    tessellation: gpd.GeoDataFrame,
    flow_color: str = const.DARK_BLUE,
    marker_color: str = const.LIGHT_BLUE,
    layer_name: str = "flows",
) -> folium.Map:
    centroids = get_centroids(tessellation)
    mean_flows = data["flow"].mean()

    def _flow_style_function(weight: float, color: str) -> Callable:
        return lambda feature: {
            "color": color,
            "weight": 5 * weight**0.5,
            "opacity": 0.65,
        }

    feature_group = folium.FeatureGroup(name=layer_name)

    origin_groups = data.groupby(by=const.ORIGIN)
    for origin, OD in origin_groups:
        lonO, latO = centroids[origin]

        for destination, flow in OD[[const.DESTINATION, const.FLOW]].values:
            lonD, latD = centroids[destination]
            gjc = LineString([(lonO, latO), (lonD, latD)])

            fgeojson = folium.GeoJson(
                gjc,
                style_function=_flow_style_function(
                    weight=flow / mean_flows, color=flow_color
                ),
            )

            popup = folium.Popup(
                f"flow from {origin} to {destination}: {int(flow)}", max_width=300
            )
            fgeojson = fgeojson.add_child(popup)
            feature_group.add_child(fgeojson)

        # Plot marker
        fmarker = folium.CircleMarker(
            [latO, lonO],
            radius=5,
            weight=2,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
        )
        T_D = [
            [destination, int(flow)]
            for destination, flow in OD[[const.DESTINATION, const.FLOW]].values
        ]
        trips_info = "<br/>".join(
            [
                f"flow to {destination}: {flow}"
                for destination, flow in sorted(T_D, reverse=True)[:5]
            ]
        )
        name = f"origin: {origin}"
        popup = folium.Popup(name + "<br/>" + trips_info, max_width=300)
        fmarker = fmarker.add_child(popup)
        feature_group.add_child(fmarker)

    feature_group.add_to(basemap)

    return basemap
