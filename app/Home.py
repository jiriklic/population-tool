"""Home page for Streamlit app."""
import os

import streamlit as st
import yaml
from src.utils import (
    add_about,
    add_logo,
    set_home_page_style,
    toggle_menu_button,
)

# Load configuration options
config_path = os.path.join(os.path.dirname(__file__), "config", "config.yml")
config = yaml.safe_load(open(config_path))

# Page configuration
st.set_page_config(layout="wide", page_title=config["browser_title"])

# If app is deployed hide menu button
toggle_menu_button()

# Create sidebar
add_logo("app/img/MA-logo.png")
add_about()

# Set page style
set_home_page_style()

# Page title
st.markdown("# Home")

# First section
st.markdown("## Introduction")
st.markdown(
    """
    <p align="justify">
        This tool enables users to extract population information for specific
        areas defined by shapefiles. This feature allows for precise and
        localized analysis of population dynamics, which is particularly useful
        for researchers and decision-makers who require population data at a
        subnational or even sub-regional level.<br><br>
        The population information is retrieved from <a href='%s'>WorldPop</a>,
        an open-access data repository that focuses on generating and providing
        detailed population data for different regions and countries around the
        world. The aim of WorldPop is to improve our understanding of human
        population dynamics and their interactions with social, economic, and
        environmental factors.<br><br>
        The organization utilizes various sources of data, including census
        records, satellite imagery, household surveys, and other demographic
        information, to estimate population distribution and demographic
        characteristics at high spatial resolutions. WorldPop employs advanced
        spatial modeling techniques and statistical methods to generate
        population estimates at fine-scale resolutions (100 meters).<br><br>
        The accuracy of the results provided
        by the tool may vary depending on the region due to limitations of the
        WorldPop's population data, in terms of spatial and temporal
        resolution, as well as inherent uncertainty resulting from modeling and
        statistical techniques used for estimation.
    </p>
"""
    % (config["url_worldpop"],),
    unsafe_allow_html=True,
)

# Second section
st.markdown("## How to use the tool")
st.markdown(
    """
    <p align="justify">
        <b>Important.</b> The duration of each step in the tool may vary
        based on the size of the uploaded data, and while the tool is
        processing, you will observe a "running" status with a runner icon
        in the top right of the screen. Please do not initiate any
        action or proceed to the next step until the current step is
        finished.<br>
    </p>
    <ul>
        <li><p align="justify">
            In the sidebar, choose <i>Population analysis</i> to start the
            analysis.
        </p>
        <li><p align="justify">
            Upload a shapefile (in .zip format) containing the areas for which
            you want to extract population information. The shapefile will be
            visualised as table underneath.
        </p>
        <li><p align="justify">
            Select the preferred type of population dataset. The options are
            <i>uncontrained</i> and <i>UNadj_contrained</i> (constrained -
            UN adjusted). For a description of these data types, please read
            the documentation.
        </p>
        <li><p align="justify">
            Select the preferred data aggregation. You can retrieve the total
            population for each area of interest, or disaggregated population
            data (by gender and age). Depending on the data type, one of the
            two options might not be available.
        </p>
        <li><p align="justify">
            Specify the year.  Total population data is available for the years
            ranging from 2000 to 2020, while disaggregated data is only
            available for the year 2020.
        </p>
        <li><p align="justify">
            If all information is filled in correctly, you will be able to
            click on the button <i>Read to run?</i>, to start the computation.
        </p>
        <li><p align="justify">
            Depending on the size of the uploaded shapefile, and on the
            complexity of the associated geometries, different methods to
            retrieve population data will be used, which results in different
            computation times. For small shapefiles (usually below 10MB), the
            computation is expected to last a few minutes, while for big
            shapefiles, the process can last up to several hours. It is
            important that you do not close the window, nor make any change
            until the process is completed.
        </p>
        <li><p align="justify">
            Once the computation is complete, the shapefile, complemented with
            population information, will be visualised.
        </p>
        <li><p align="justify">
            It is possible to download the output in shapefile format or csv
            format.
        </p>
        <li><p align="justify">
            Select the shapefile field you want to use to assign labels to the
            population data in the plot.
        </p>
        <li><p align="justify">
            Create the plot, by clicking on the button.
        </p>
        <li><p align="justify">
            The interactive plot shows population information for each area
            of interest. If disaggregated data was selected, a population
            pyramid will be created for each area.
        </p>
        <li><p align="justify">
            By clicking on <i>Download figure(s)</i>, you can download the
            plots. The plots are saved as .png figures. If disaggregated data
            was selected, a .zip file is created, containing one figure for
            each area of interest.
        </p>
        <li><p align="justify">
            You can restart the analysis by clicking on the correponding
            button.
        </p>
    </ul>
    <p align="justify">
        In case you get errors, follow the intructions, or try to refresh the
        web page. If you have doubts, feel free to contact the Data Science
        team.
    </p>
    """,
    unsafe_allow_html=True,
)
