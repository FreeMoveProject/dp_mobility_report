import math
from typing import Optional, Tuple, Type, Union

import folium
import matplotlib
import matplotlib as mpl
import numpy as np
import seaborn as sns
from geopandas import GeoDataFrame
from matplotlib import pyplot as plt
from pandas import DataFrame

sns.set_theme()
dark_blue = "#283377"
light_blue = "#5D6FFF"
orange = "#D9642C"
light_orange = "#FFAD6F"
grey = "#8A8A8A"
light_grey = "##f2f2f2"


def format(value: Union[float, int], type: Type, ndigits: int = 2) -> Union[float, int]:
    value = type(value)
    if type == float:
        value = round(value, ndigits)
    return value


def histogram(
    hist: Tuple,
    x_axis_label: str,
    y_axis_label: str = "Frequency",
    margin_of_error: float = None,
    rotate_label: bool = False,
    x_axis_type: Type = float,
    ndigits_x_label: int = 2,
) -> mpl.figure.Figure:
    bins = hist[1]
    counts = hist[0]

    # single integers (instead of bin ranges) as x axis labels
    if len(bins) == len(counts):
        labels = np.array([format(bin, x_axis_type) for bin in bins[:-1]])
        if bins[-1] == np.Inf:
            labels = np.append(labels, f"> {labels[-1]}")
        else:
            labels = np.append(labels, format(bins[-1], x_axis_type))

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

    return barchart(
        labels,
        counts,
        x_axis_label,
        y_axis_label,
        margin_of_error=margin_of_error,
        rotate_label=rotate_label,
    )


def barchart(
    x: np.ndarray,
    y: np.ndarray,
    x_axis_label: str,
    y_axis_label: str,
    margin_of_error: Optional[float] = None,
    rotate_label: bool = False,
) -> mpl.figure.Figure:
    fig, ax = plt.subplots()
    ax.bar(x, y, yerr=margin_of_error, align="center", alpha=0.5, capsize=10)
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
    simulations: list = None,
    margin_of_error: float = None,
    add_diagonal: bool = False,
    rotate_label: bool = False,
) -> mpl.figure.Figure:
    fig, ax = plt.subplots()
    if simulations is not None:
        ax.plot(data[x], data[simulations], light_grey)
    if margin_of_error is not None:
        ax.fill_between(
            data[x],
            (data[y] - margin_of_error),
            (data[y] + margin_of_error),
            color="blue",
            alpha=0.1,
        )
    ax.plot(data[x], data[y])
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel(x_axis_label)
    ax.set_ylim(bottom=0)
    if add_diagonal:
        ax.plot([0, data[x].max()], [0, data[y].max()], grey)

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
    hue_order: Optional[list] = None,
    margin_of_error: Optional[float] = None,
) -> mpl.figure.Figure:
    fig = plt.figure()
    plot = fig.add_subplot(111)
    palette = [dark_blue, light_blue, orange, light_orange]

    sns.lineplot(
        data=data,
        x=x,
        y=y,
        hue=color,
        ax=plot,
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
    min_scale: Optional[Union[int, float]] = None,
    aliases: list = None,
) -> folium.Map:
    poly_json = counts_per_tile_gdf.to_json()

    center_x, center_y = _get_center(counts_per_tile_gdf)
    m = _basemap(center_x, center_y)
    min_scale = (
        counts_per_tile_gdf[fill_color_name].min() if min_scale is None else min_scale
    )

    # color
    cmap = mpl.cm.viridis_r
    norm = mpl.colors.Normalize(
        vmin=min_scale, vmax=counts_per_tile_gdf[fill_color_name].max()
    )

    def _hex_color(x: Union[float, int]) -> str:
        rgb = norm(x)
        return matplotlib.colors.rgb2hex(cmap(rgb))

    def _get_color(x: dict, fill_color_name: str) -> str:
        if x["properties"][fill_color_name] is None:
            return "#8c8c8c"
        return _hex_color(x["properties"][fill_color_name])

    def _style_function(x: dict) -> dict:
        return {
            "fillColor": _get_color(x, fill_color_name),
            "color": grey,
            "weight": 1.5,
            "fillOpacity": 0.6,
        }

    if "tile_name" in counts_per_tile_gdf:
        fields = ["tile_id", "tile_name", fill_color_name]
    else:
        fields = ["tile_id", fill_color_name]
    folium.GeoJson(
        poly_json,
        style_function=_style_function,
        popup=folium.GeoJsonPopup(fields=fields, aliases=aliases),
    ).add_to(m)

    # colorbar object to create custom legend
    colorbar, ax = plt.subplots(figsize=(6, 1))
    colorbar.subplots_adjust(bottom=0.5)
    colorbar.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="horizontal",
        label=scale_title,
    )
    # TODO: add legend directly to map

    return m, colorbar


def multi_choropleth_map(
    counts_per_tile_timewindow: DataFrame,
    tessellation: GeoDataFrame,
    diverging_cmap: bool = False,
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
    vmin = counts_per_tile_timewindow.iloc[:, 2:].min().min()
    vmin = vmin if vmin is not np.nan else 0
    vmax = counts_per_tile_timewindow.iloc[:, 2:].max().max()
    vmax = vmax if vmax is not np.nan else 2

    # color
    if diverging_cmap:
        vmin = 0
        vmax = 2 if vmax <= 1 else vmax
        cmap = "RdBu_r"
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=1, vmax=vmax)
    else:
        cmap = "viridis_r"
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

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
                    "color": "lightgrey",
                },
            )
            ax.set_title(column_name)

    # Create colorbar as a legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []  # add the colorbar to the figure
    # set the range for the choropleth
    fig.colorbar(sm, ax=axes)
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
    x: np.ndarray,
    x_axis_label: str,
    y_labels: list = None,
    margin_of_error: Optional[float] = None,
) -> mpl.figure.Figure:
    y = list(range(1, len(x) + 1))[::-1]
    y_labels = y if y_labels is None else y_labels
    fig, ax = plt.subplots()
    ax.errorbar(
        x,
        y,
        xerr=margin_of_error,
        fmt="o",
        ecolor="lightblue",
        elinewidth=5,
        capsize=10,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel("Rank")
    ax.set_xlabel(x_axis_label)
    return fig
