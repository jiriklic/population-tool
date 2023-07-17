"""Functions for plotting."""
import copy
import io
import math
import tempfile
import urllib.request
from typing import List, Optional, Union
from zipfile import ZipFile

import folium
import geopandas as gpd
import matplotlib
import numpy as np
import plotly.graph_objs as go
import shapely
import streamlit_ext as ste
from plotly.subplots import make_subplots


def display_polygons_on_map(
    data: Union[gpd.GeoDataFrame, shapely.Polygon],
    add_countryborders: bool = True,
    high_res: bool = False,
    world_gdf: Optional[gpd.GeoDataFrame] = None,
) -> folium.Map:
    """
    Display polygons on a map.

    If not given, country borders are retrieved from NaturalEarth
    (10m resolution).

    Inputs
    -------
    data (geopandas.GeoDataFrame or shapely.Polygon): GeoDataFrame containing
        Polygon geometries or Polygon.
    add_countryborders (bool, optional): if True, draw country borders. Default
        to True.
    high_res (bool, optional): if True, retrieve country borders at high
        resolution (1:10m instead of 1:110m). Used only if add_countryborders
        is True and world_gdf is None.
    world_gdf (geopandas.GeoDataFrame, optional): world countries geometries.
        If None, country borders will be retrieved from NaturalEarth (10m
        resolution).

    Returns
    -------
    m (folium.Map): folium Map with geometries.
    """
    if add_countryborders and world_gdf is None:
        if high_res:
            link = (
                "https://www.naturalearthdata.com/http//www.naturalearthdata."
                "com/"
                "download/10m/cultural/ne_10m_admin_0_countries.zip"
            )
            header = (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 "
                "Safari/537.36"
            )
            req = urllib.request.Request(link, headers={"User-Agent": header})
            world_gdf = gpd.read_file(urllib.request.urlopen(req))
        else:
            world_gdf = gpd.read_file(
                gpd.datasets.get_path("naturalearth_lowres")
            )

    m = folium.Map(
        zoom_start=10,
        control_scale=True,
    )

    # restrict maps to boundaries geodataframe
    if type(data) == gpd.geodataframe.GeoDataFrame:
        bounds = data.total_bounds
    else:
        bounds = np.array(data.bounds)
    m.fit_bounds([bounds[:2].tolist()[::-1], bounds[2:].tolist()[::-1]])

    # add country borders as a GeoJSON layer
    if add_countryborders:
        country_borders = world_gdf
        folium.GeoJson(
            country_borders,
            style_function=lambda feature: {
                "fillColor": "white",
                "color": "black",
                "weight": 1,
                "fillOpacity": 0,
            },
        ).add_to(m)

    folium.GeoJson(
        data,
        style_function=lambda feature: {
            "fillColor": "blue",
            "color": "red",
            "weight": 1,
            "fillOpacity": 0.5,
        },
    ).add_to(m)

    return m


def color_list_from_cmap(cmap_name: str, n: int) -> Union[List[str], str]:
    """
    Generate a list of n colors from a specified color map.

    Inputs
    -------
    cmap_name (str): name of the color map to use. This should be a valid
    color map name recognized by matplotlib.cm.get_cmap().

    n (int): number of colors to generate in the color list.

    Returns
    -------
    color_list (list of str): list of n colors represented as hexadecimal
        strings.
    """
    cmap = matplotlib.cm.get_cmap(cmap_name)
    color_list = cmap(np.array(np.rint(np.linspace(0, cmap.N, n)), dtype=int))
    return [matplotlib.colors.to_hex(c) for c in color_list]


def format_tick_labels(x_list: list) -> List[str]:
    """
    Format list of ticklabels for plotting.

    Inputs
    -------
    x_list (str): list of floats or ints

    Returns
    -------
    List(str): updated list of strings
    """
    if max(x_list) >= 1e6:
        return [f"{int(x/1e6)}M" if x != 0 else "0" for x in x_list]
    else:
        return x_list


def plot_pop_data(
    gdf: gpd.GeoDataFrame,
    col_label: str,
    legend_title: str,
    aggregated: bool = True,
    stacked: bool = False,
    cmap_name: str = "viridis",
) -> go.Figure:
    """
    Plot population data on a bar chart.

    Inputs
    -------
    gdf (geopandas.GeoDataFrame): GeoDataFrame object containing geographic
        data.
    col_label (str): column in the GeoDataFrame that contains the labels
        for plotting.
    legend_title (str): title of the legend for the plot (when
        plotting disaggregated stacked data) or the label of the y-axis (when
        plotting aggregated data).
    aggregated (bool, optional): boolean indicating whether the population
        data aggregated or not. Default to True.
    stacked (bool, optional): boolean indicating whether a stacked barplot
        should be created. This parameter is only used if aggregated is False.
        Default to False.
    cmap_name (str, optional): name of the color map to be used for the
        plot. Default to "viridis".

    Returns
    -------
    fig (plotly.graph_objs.Figure): plotly figure object containing the plot.

    """
    if aggregated:
        return plot_pop_data_aggregated(
            gdf=gdf,
            col_label=col_label,
            label_title=legend_title,
            cmap_name=cmap_name,
        )
    else:
        return plot_pop_data_disaggregated(
            gdf=gdf,
            col_label=col_label,
            legend_title=legend_title,
            stacked=stacked,
            cmap_name=cmap_name,
        )


def plot_pop_data_aggregated(
    gdf: gpd.GeoDataFrame,
    col_label: str,
    label_title: str,
    cmap_name: str = "viridis",
) -> go.Figure:
    """
    Plot aggregated population data on a bar chart.

    Inputs
    -------
    gdf (geopandas.GeoDataFrame): GeoDataFrame object containing geographic
        data.
    col_label (str): column in the GeoDataFrame that contains the labels
        for plotting.
    label_title (str): label for the y-axis.
    cmap_name (str, optional): name of the color map to be used for the
        plot. Default to "viridis".

    Returns
    -------
    fig (plotly.graph_objs.Figure): plotly figure object containing the plot.

    """
    y = gdf[col_label].tolist()
    colors = color_list_from_cmap(cmap_name, len(gdf))[::-1]

    fig = go.Figure(
        data=[
            go.Bar(
                y=y,
                x=gdf["pop_tot"].tolist(),
                orientation="h",
                hoverinfo="x",
                marker_color=colors,
            )
        ]
    )

    fig.update_layout(
        yaxis={
            "title": label_title,
            "type": "category",
            "categoryorder": "total descending",
        },
        xaxis={"title": "Population"},
        margin=dict(pad=5),
    )

    fig.update_yaxes(title_standoff=20)
    fig.update_xaxes(title_standoff=20)

    return fig


def plot_pop_data_disaggregated(
    gdf: gpd.GeoDataFrame,
    col_label: str,
    legend_title: str,
    stacked: bool = False,
    cmap_name: str = "viridis",
) -> go.Figure:
    """
    Plot disaggregated population data on a bar chart.

    Inputs
    -------
    gdf (geopandas.GeoDataFrame): GeoDataFrame object containing geographic
        data.
    col_label (str): column in the GeoDataFrame that contains the labels
        for plotting.
    legend_title (str): title of the legend for the plot.
    stacked (bool, optional): boolean indicating whether a stacked barplot
        should be created. Default to False.
    cmap_name (str, optional): name of the color map to be used for the
        plot. Default to "viridis".

    Returns
    -------
    fig (plotly.graph_objs.Figure): plotly figure object containing the plot.

    """
    if stacked:
        return plot_pop_data_stacked(
            gdf=gdf,
            col_label=col_label,
            legend_title=legend_title,
            cmap_name=cmap_name,
        )
    else:
        return plot_pop_data_split(
            gdf=gdf,
            col_label=col_label,
        )


def plot_pop_data_split(
    gdf: gpd.GeoDataFrame,
    col_label: str,
) -> go.Figure:
    """
    Plot disaggregated non-stacked population data on a bar chart.

    One plot per gender/age will be visualised.

    Inputs
    -------
    gdf (geopandas.GeoDataFrame): GeoDataFrame object containing geographic
        data.
    col_label (str): column in the GeoDataFrame that contains the labels
        for plotting.

    Returns
    -------
    fig (plotly.graph_objs.Figure): plotly figure object containing the plot.

    """
    data_pop = gdf[[col for col in gdf.columns if "pop_m" in col]].values
    data_max = data_pop.max()

    order_or_mag = math.floor(math.log(data_max, 10))
    axis_max = (
        math.ceil(data_max * 2 / (10**order_or_mag)) / 2 * 10**order_or_mag
    )
    ticks = list(
        np.arange(
            0,
            axis_max + 0.5 * (10**order_or_mag),
            0.5 * (10**order_or_mag),
            dtype=int,
        )
    )

    y = [int(col[-2:]) for col in gdf.columns if "pop_m" in col]

    fig = go.Figure()

    for i in range(len(gdf)):
        visible = True if i == 0 else False
        fig = fig.add_trace(
            go.Bar(
                y=y,
                x=gdf.iloc[i][
                    [col for col in gdf.columns if "pop_m" in col]
                ].values,
                meta=gdf[col_label].tolist()[i],
                hovertext=gdf.iloc[i][
                    [col for col in gdf.columns if "pop_m" in col]
                ].values,
                hoverinfo="text",
                orientation="h",
                name="Men",
                marker=dict(color="powderblue"),
                width=3,
                visible=visible,
            )
        )
        fig = fig.add_trace(
            go.Bar(
                y=y,
                x=gdf.iloc[i][
                    [col for col in gdf.columns if "pop_f" in col]
                ].values
                * -1,
                meta=gdf[col_label].tolist()[i],
                hovertext=gdf.iloc[i][
                    [col for col in gdf.columns if "pop_f" in col]
                ].values,
                hoverinfo="text",
                orientation="h",
                name="Women",
                marker=dict(color="seagreen"),
                width=3,
                visible=visible,
            )
        )

    update_dict_list = [
        dict(
            label=gdf[col_label].tolist()[i],
            method="update",
            args=[
                {
                    "visible": [
                        True if k == i * 2 or k == i * 2 + 1 else False
                        for k in range(2 * len(gdf))
                    ]
                },
                {
                    "title": f"<b>{gdf[col_label].tolist()[i]}",
                    "showlegend": True,
                },
            ],
        )
        for i in range(len(gdf))
    ]

    fig = fig.update_xaxes(
        range=[-axis_max, axis_max],
        showline=True,
        linecolor="#000",
        tickvals=[-t for t in ticks if t != 0][::-1] + ticks,
        ticktext=format_tick_labels(
            [t for t in ticks if t != 0][::-1] + ticks
        ),
    )

    fig = fig.update_layout(
        barmode="overlay",
        template="ggplot2",
        updatemenus=[
            go.layout.Updatemenu(
                active=0,
                buttons=list(update_dict_list),
                direction="up",
                x=1.05,
                xanchor="left",
                y=0.05,
                yanchor="bottom",
                font=dict(size=11),
            )
        ],
    )

    fig.update_layout(title_text=f"<b>{gdf[col_label].tolist()[0]}")

    return fig


def plot_pop_data_stacked(
    gdf: gpd.GeoDataFrame,
    col_label: str,
    legend_title: str,
    cmap_name: str = "viridis",
) -> go.Figure:
    """
    Plot disaggregated stacked population data on a bar chart.

    Inputs
    -------
    gdf (geopandas.GeoDataFrame): GeoDataFrame object containing geographic
        data.
    col_label (str): column in the GeoDataFrame that contains the labels
        for plotting.
    legend_title (str): title of the legend for the plot.
    cmap_name (str, optional): name of the color map to be used for the
        plot. Default to "viridis".

    Returns
    -------
    fig (plotly.graph_objs.Figure): plotly figure object containing the plot.

    """
    data_max = max(
        gdf.filter(like="pop_m_").sum().tolist()
        + gdf.filter(like="pop_f_").sum().tolist()
    )
    order_or_mag = math.floor(math.log(data_max, 10))
    axis_max = math.ceil(data_max / (10**order_or_mag)) * 10**order_or_mag
    ticks = list(
        np.arange(
            0,
            axis_max + 0.5 * (10**order_or_mag),
            0.5 * (10**order_or_mag),
            dtype=int,
        )
    )

    y = [int(col[-2:]) for col in gdf.columns if "pop_m" in col]
    colors = color_list_from_cmap(cmap_name, len(gdf))[::-1]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["<b>Women", "<b>Men"],
        shared_yaxes=True,
        horizontal_spacing=0,
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
    )

    for i in range(len(gdf)):
        fig.add_trace(
            go.Bar(
                y=y,
                x=gdf.iloc[i][
                    [col for col in gdf.columns if "pop_f" in col]
                ].values
                * -1,
                hovertext=gdf.iloc[i][
                    [col for col in gdf.columns if "pop_f" in col]
                ].values,
                hoverinfo="text",
                orientation="h",
                marker=dict(color=colors[i]),
                width=3,
                showlegend=False,
            ),
            secondary_y=False,
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                y=y,
                x=gdf.iloc[i][
                    [col for col in gdf.columns if "pop_m" in col]
                ].values,
                hovertext=gdf.iloc[i][
                    [col for col in gdf.columns if "pop_m" in col]
                ].values,
                hoverinfo="text",
                orientation="h",
                marker=dict(color=colors[i]),
                width=3,
                name=gdf[col_label].tolist()[i],
            ),
            secondary_y=True,
            row=1,
            col=2,
        )

    fig = fig.update_xaxes(
        showline=False,
        linecolor="#000",
        tickvals=[-t for t in ticks if t != 0][::-1] + ticks,
        ticktext=format_tick_labels(
            [t for t in ticks if t != 0][::-1] + ticks
        ),
    )

    fig.layout.yaxis4.update(ticks="", showticklabels=False)

    fig.update_layout(
        barmode="stack",
        template="ggplot2",
        legend={
            "title": f"<b>{legend_title}</b>" + "\n\n",
            "traceorder": "normal",
        },
    )

    return fig


def save_pngs_with_bytesio(
    fig_list: List[go.Figure],
    directory: str,
) -> str:
    """
    Create zipped file containing list of figures saved as png.

    A zipped shapefile, as well as figures with format .png, are saved
    in the folder defined by the argument `directory`.

    Inputs
    -------
    fig_list (list): list of plotly figures.
    directory (str): filepath where the files are saved.

    Returns
    -------
    zip_filename (str): filename of the zip file.
    """
    zip_filename = "user_figures_zip.zip"
    zipObj = ZipFile(f"{directory}/{zip_filename}", "w")

    for i, fig in enumerate(fig_list):
        fig.write_image(file=f"{directory}/fig_{i}.png")

        zipObj.write(f"{directory}/fig_{i}.png", arcname=f"fig_{i}.png")

    zipObj.close()

    return zip_filename


def st_download_figures(
    fig: go.Figure,
    gdf: gpd.GeoDataFrame,
    aggregated: bool,
    stacked: bool,
    col_label: str,
    filename: str,
    label: str = "Download figure(s)",
) -> None:
    """
    Create a button to download figures with Streamlit.

    If there are several snapshots in the same figure, each of them is saved as
    a .png figure in a zipped file.

    Inputs
    -------
    fig (go.Figure): input figure.
    gdf (geopandas.GeoDataFrame): input GeoDataFrame.
    aggregated (bool): True if data is aggregated (by gender and age).
    stacked (bool): True if disaggregated data should be visualised as stacked
        pyramid, False otherwise.
    col_label (str): column in the GeoDataFrame that contains the labels
        for plotting. Here it is used for the title and to retrieve the
        snapshots of the figure.
    filename (str): name of the saved file.
    label (str, optional): button label. Default to "Download shapefile".
    """
    with tempfile.TemporaryDirectory() as tmp:
        # In this case there are several figures
        if not aggregated and not stacked:
            fig_list = []
            for i in range(len(gdf)):
                fig_i = copy.deepcopy(fig)
                fig_i.update_layout(
                    updatemenus=[
                        go.layout.Updatemenu(active=i, visible=False)
                    ],
                    title=f"<b>{gdf[col_label].tolist()[i]}",
                )
                fig_i.for_each_trace(
                    lambda trace: trace.update(visible=True)
                    if trace.meta == gdf[col_label].tolist()[i]
                    else trace.update(visible=False),
                )
                fig_list.append(fig_i)
            # create the shape files in the temporary directory
            zip_filename = save_pngs_with_bytesio(fig_list, tmp)
            with open(f"{tmp}/{zip_filename}", "rb") as file:
                ste.download_button(
                    label=label,
                    data=file,
                    file_name=f"{filename}.zip",
                    mime="application/zip",
                )
        # In this case there is only one figure
        else:
            # Create an in-memory buffer
            buffer = io.BytesIO()

            # Save the figure as a pdf to the buffer
            fig.write_image(file=buffer, format="png")

            # Download the pdf from the buffer
            ste.download_button(
                label=label,
                data=buffer,
                file_name=f"{filename}.png",
                mime="image/png",
            )
