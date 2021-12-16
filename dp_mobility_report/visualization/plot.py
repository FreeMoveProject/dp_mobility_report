"""Plot functions for the profiling report."""
import math

import folium
import matplotlib as mpl
import numpy as np
import seaborn as sns
from branca.colormap import linear
from matplotlib import pyplot as plt

sns.set_theme()
dark_blue = "#283377"
light_blue = "#5D6FFF"
orange = "#D9642C"
light_orange = "#FFAD6F"
grey = "#8A8A8A"


def histogram(hist, x_axis_label=None, rotate_label=False, x_axis_type=None):
    bins = hist[1]
    if x_axis_type is not None:
        bins = bins.astype(x_axis_type)
    lower_bound = bins[:-1]
    upper_bound = bins[1:]
    labels = np.array(
        [
            "[" + str(x1) + "\n - \n " + str(x2) + ")"
            for x1, x2 in zip(lower_bound, upper_bound)
        ]
    )
    counts = hist[0]
    return barchart(
        labels, counts, x_axis_label, "Frequency", rotate_label=rotate_label
    )


def barchart(
    x, y, x_axis_label, y_axis_label, data=None, order_x=None, rotate_label=False
):
    fig = plt.figure()
    plot = fig.add_subplot(111)
    sns.barplot(data=data, x=x, y=y, color=dark_blue, ax=plot, order=order_x)
    plot.set_ylabel(y_axis_label)
    plot.set_xlabel(x_axis_label)

    if rotate_label:
        plt.xticks(rotation=90)
    return fig


def linechart(data, x, y, x_axis_label, y_axis_label, add_diagonal=False):
    fig = plt.figure()
    plot = fig.add_subplot(111)
    sns.lineplot(data=data, x=x, y=y, ax=plot)
    plot.set_ylabel(y_axis_label)
    plot.set_xlabel(x_axis_label)
    plot.set_ylim(bottom=0)

    if add_diagonal:
        plot.plot([0, data[x].max()], [0, data[y].max()], grey)
    return fig


def multi_linechart(data, x, y, color, hue_order=None):
    fig = plt.figure()
    plot = fig.add_subplot(111)
    sns.lineplot(
        data=data,
        x=x,
        y=y,
        hue=color,
        ax=plot,
        hue_order=hue_order,
        palette=[dark_blue, light_blue, orange, light_orange],
    )
    plot.set_ylim(bottom=0)
    return fig


def choropleth_map(
    counts_per_tile_gdf, fill_color_name, scale_title="Visit count", min_scale=None
):
    poly_json = counts_per_tile_gdf.to_json()

    center_x, center_y = _get_center(counts_per_tile_gdf)
    m = _basemap(center_x, center_y)
    min_scale = (
        counts_per_tile_gdf[fill_color_name].min() if min_scale is None else min_scale
    )
    color_map = linear.YlGnBu_09.scale(
        min_scale, counts_per_tile_gdf[fill_color_name].max()
    )

    def get_color(x, fill_color_name):
        if x["properties"][fill_color_name] is None:
            return "#8c8c8c"
        return color_map(x["properties"][fill_color_name])

    style_function = lambda x: {
        "fillColor": get_color(x, fill_color_name),
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
        style_function=style_function,
        popup=folium.GeoJsonPopup(fields=fields),
    ).add_to(m)

    color_map.caption = scale_title
    color_map.add_to(m, name=fill_color_name)
    return m


# TODO: adjust color
def multi_choropleth_map(counts_per_tile_timewindow, tessellation):
    counts_per_tile_timewindow = tessellation[["tile_id", "geometry"]].merge(
        counts_per_tile_timewindow, left_on="tile_id", right_index=True, how="left"
    )

    # col1: tile_id, col2: geometry
    plot_count = counts_per_tile_timewindow.shape[1] - 2
    plots_per_row = 3
    row_count = math.ceil(plot_count / plots_per_row)
    fig, axes = plt.subplots(row_count, plots_per_row, figsize=(15, 10))

    # upper and lower bound
    vmin = counts_per_tile_timewindow.iloc[:, 2:].min().min()
    vmax = counts_per_tile_timewindow.iloc[:, 2:].max().max()
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
                cmap="viridis_r",
                norm=norm,
                linewidth=0.1,
                ax=ax,
                edgecolor="#FFFFFF",
                missing_kwds=dict(
                    color="lightgrey",
                ),
            )
            ax.set_title(column_name)

    # Create colorbar as a legend
    sm = plt.cm.ScalarMappable(
        cmap="viridis_r", norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm._A = []  # add the colorbar to the figure
    # set the range for the choropleth
    fig.colorbar(sm, ax=axes)
    return fig


def _get_center(gdf):
    center_x = (gdf.total_bounds[0] + gdf.total_bounds[2]) / 2
    center_y = (gdf.total_bounds[1] + gdf.total_bounds[3]) / 2
    return (center_x, center_y)


def _basemap(center_x, center_y, zoom_start=10):
    return folium.Map(
        tiles="CartoDB positron",
        attr='&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>, &copy; <a href="https://carto.com/attributions">CARTO</a>',
        zoom_start=zoom_start,
        location=[center_y, center_x],
        control_scale=True,
    )
