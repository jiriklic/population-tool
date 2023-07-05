"""Proximity analysis page for Streamlit app."""
import time

import streamlit as st
from src.utils import (
    add_about,
    add_logo,
    elapsed_time_string,
    set_tool_page_style,
    toggle_menu_button,
)
from src.utils_ee import ee_initialize
from src.utils_plotting import plot_pop_data, st_download_figures
from src.utils_population import (
    add_population_data,
    load_gdf,
    st_download_shapefile,
    visualize_data,
)

# Page configuration
# st.set_page_config(layout="wide", page_title=config["browser_title"])

# If app is deployed hide menu button
toggle_menu_button()

# Create sidebar
add_logo("app/img/MA-logo.png")
add_about()

# Page title
st.markdown("# Population analysis")

# Set page style
set_tool_page_style()

# Initialise Google Earth Engine
ee_initialize()

# Parameters
verbose = True
limit_size_data = False

# Set initial stage to 0
if "stage" not in st.session_state:
    st.session_state.stage = -1

if "gdf_with_pop" not in st.session_state:
    st.session_state.gdf_with_pop = None

aggregation_default_options = [
    "Total population",
    "Disaggregated population (gender, age)",
]
aggregation_dict = {
    aggregation_default_options[0]: True,
    aggregation_default_options[1]: False,
}


# Define function to change stage
def set_stage(stage):
    """
    Set stage for the app.

    Each time a certain button is pressed, the stage progresses by one.
    """
    st.session_state.stage = stage


# Create file uploader object for POIs
upload_geometry_file = st.file_uploader(
    "Upload a zipped shapefile containing your geometries",
    type=["zip"],
    on_change=set_stage,
    args=(0,),
)
print(
    type(upload_geometry_file) == st.runtime.uploaded_file_manager.UploadedFile
)

if not upload_geometry_file:
    st.session_state.stage = -1

if st.session_state.stage >= 0:
    gdf, size_gdf, error = load_gdf(upload_geometry_file)

    if error:
        st.error(error)
        st.stop()

    # TODO: creating folium maps takes time, this should not be run every time
    # st.cache does not work, as there are no outputs
    # st.form does not work, as there are several buttons and callbacks in
    # the app that are not compatible with it
    visualize_data(gdf)

    options_default = ["", "unconstrained", "UNadj_constrained"]
    data_type = st.selectbox(
        "Select type of population dataset",
        options=options_default,
        index=0,
        on_change=set_stage,
        args=(1,),
    )

if st.session_state.stage >= 1:
    if data_type == "":
        aggregation_options = [""]
    elif data_type == "unconstrained":
        aggregation_options = [""] + aggregation_default_options
    else:
        aggregation_options = ["", aggregation_default_options[1]]
    aggregation = st.selectbox(
        "Select the data aggregation",
        options=aggregation_options,
        index=0,
        on_change=set_stage,
        args=(2,),
    )

if st.session_state.stage >= 2:
    if aggregation == "":
        year_options = [""]
    elif aggregation == aggregation_default_options[0]:
        year_options = [""] + [str(y) for y in range(2000, 2021)]
    else:
        year_options = [""] + ["2020"]
    year = st.selectbox(
        "Select year",
        options=year_options,
        index=0,
        on_change=set_stage,
        args=(3,),
    )

    st.markdown("### ")

if st.session_state.stage >= 3:
    if all([param != "" for param in [year, aggregation, data_type]]):
        run = st.button("Ready to run?", on_click=set_stage, args=(4,))


# Run computations
if st.session_state.stage >= 4:
    st.markdown("""---""")
    st.markdown("## Results")

    # run analysis
    if run:
        if limit_size_data:
            gdf = gdf.iloc[:4,]
        start_time = time.time()
        gdf_with_pop = add_population_data(
            gdf=gdf,
            size_gdf=size_gdf,
            data_type=data_type,
            year=int(year),
            aggregated=aggregation_dict[aggregation],
            progress_bar=True,
        )
        elapsed = time.time() - start_time
        st.markdown(elapsed_time_string(elapsed))
        st.session_state.gdf_with_pop = gdf_with_pop

    st.success("Computation complete.")
    st.dataframe(st.session_state.gdf_with_pop.to_wkt())

    st_download_shapefile(
        gdf=st.session_state.gdf_with_pop,
        filename="Shapefile_with_pop_data.zip",
        label="Download shapefile",
    )

    st.markdown("""---""")
    st.markdown("## Population plots")

    col_label = st.selectbox(
        "Select field for label",
        options=st.session_state.gdf_with_pop.columns.to_list(),
        index=0,
        on_change=set_stage,
        args=(4,),
    )

    plot_button = st.button("Create plot", on_click=set_stage, args=(5,))


if st.session_state.stage >= 5:
    legend_title = ""
    plot_title = "Name geometry"
    joint = False

    fig = plot_pop_data(
        gdf=st.session_state.gdf_with_pop,
        joint=joint,
        col_label=col_label,
        legend_title=legend_title,
        plot_title=plot_title,
        aggregated=aggregation_dict[aggregation],
    )

    st.plotly_chart(fig, use_container_width=True)

    st_download_figures(
        fig=fig,
        gdf=st.session_state.gdf_with_pop,
        col_label=col_label,
        filename="Figure_pop_data",
        label="Download figure(s)",
        aggregated=aggregation_dict[aggregation],
    )

    st.button("Reset analysis", on_click=set_stage, args=(0,))
