"""Proximity analysis page for Streamlit app."""
import geopandas as gpd
import streamlit as st
from src.config_parameters import params
from src.utils import (
    add_about,
    add_logo,
    set_tool_page_style,
    toggle_menu_button,
)
from src.utils_plotting import plot_pop_data
from src.utils_population import add_population_data, check_gdf_geometry

# Page configuration
st.set_page_config(layout="wide", page_title=params["browser_title"])

# If app is deployed hide menu button
toggle_menu_button()

# Create sidebar
add_logo("app/img/MA-logo.png")
add_about()

# Page title
st.markdown("# Population analysis")

# Set page style
set_tool_page_style()

# Parameters
verbose = True
limit_size_data = False

# Set initial stage to 0
if "stage" not in st.session_state:
    st.session_state.stage = 0

if "gdf_with_pop" not in st.session_state:
    st.session_state.gdf_with_pop = None

if "data_type" not in st.session_state:
    st.session_state.data_type = ""

aggregation_options = [
    "Total population",
    "Disaggregated population (gender, age)",
]
aggregation_dict = {
    aggregation_options[0]: True,
    aggregation_options[1]: False,
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
    args=(1,),
)

if not upload_geometry_file:
    st.session_state.stage = 0

if st.session_state.stage > 0:
    try:
        gdf = gpd.read_file(upload_geometry_file)
        if not check_gdf_geometry(gdf):
            st.error(
                "The  shapefile contains geometries that are not "
                "polygons. Check the source data."
            )
            st.stop()

    except Exception:
        st.error("Error with importing the shapefile. Check the source data.")
        st.stop()

    st.dataframe(gdf.to_wkt())

    options_default = ["unconstrained", "constrained", "UNadj_constrained"]
    if st.session_state.data_type not in options_default:
        options_default.append(st.session_state.data_type)
    index = options_default.index(st.session_state.data_type)
    data_type = st.selectbox(
        "Select type of population dataset",
        options=options_default,
        index=index,
        on_change=set_stage,
        args=(2,),
        key="data_type",
    )

if st.session_state.stage > 1:
    year_options = (
        list(range(2000, 2021))
        if st.session_state.data_type == "unconstrained"
        else [2020]
    )
    year = st.selectbox(
        "Select year",
        options=year_options,
        index=len(year_options) - 1,
        on_change=set_stage,
        args=(2,),
    )
    aggregation = st.selectbox(
        "Select the data aggregation",
        options=aggregation_options,
        index=0,
        on_change=set_stage,
        args=(2,),
    )
    st.markdown("### ")
    run = st.button("Ready to run?", on_click=set_stage, args=(3,))


# Run computations
if st.session_state.stage > 2:
    st.markdown("""---""")
    st.markdown("## Results")

    # run analysis
    if run:
        if limit_size_data:
            gdf = gdf.iloc[:4,]
        gdf_with_pop = add_population_data(
            gdf=gdf,
            data_type=st.session_state.data_type,
            year=year,
            aggregated=aggregation_dict[aggregation],
            progress_bar=True,
        )
        st.session_state.gdf_with_pop = gdf_with_pop

    st.success("Computation complete.")
    st.dataframe(st.session_state.gdf_with_pop.to_wkt())

    st.markdown("""---""")
    st.markdown("## Population plots")

    col_label = st.selectbox(
        "Select field for label",
        options=st.session_state.gdf_with_pop.columns.to_list(),
        index=0,
        on_change=set_stage,
        args=(4,),
    )

    plot_button = st.button("Create plot", on_click=set_stage, args=(4,))


if st.session_state.stage > 3:
    legend_title = ""
    plot_title = "Name geometry"
    joint = False

    fig = plot_pop_data(
        gdf=st.session_state.gdf_with_pop,
        joint=joint,
        col_label=col_label,
        legend_title=legend_title,
        plot_title=plot_title,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.button("Reset analysis", on_click=set_stage, args=(0,))
