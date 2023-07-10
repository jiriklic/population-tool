"""Documentation page for Streamlit app."""
import os

import streamlit as st
import yaml
from PIL import Image
from src.utils import (
    add_about,
    add_logo,
    set_doc_page_style,
    toggle_menu_button,
)

# Load configuration options
config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "config", "config.yml")
)
config = yaml.safe_load(open(config_path))

# Page configuration
# st.set_page_config(layout="wide", page_title=config["browser_title"])

# If app is deployed hide menu button
toggle_menu_button()

# Create sidebar
add_logo("app/img/MA-logo.png")
add_about()

# Set page style
set_doc_page_style()

# Page title
st.markdown("# Documentation")

# Methodology section
st.markdown("## Methodology")
st.markdown(
    """
    <h3>Worldpop data</h3>
    <p align="justify">
        WorldPop is a renowned research initiative and open-access data
        repository that plays a pivotal role in providing comprehensive and
        detailed population data for regions and countries around the world.
        Their extensive datasets offer valuable insights into population
        dynamics and their interactions with various socio-economic and
        environmental factors.<br><br>
        Available datasets encompass both population totals and disaggregated
        information, enabling researchers and practitioners to analyze
        population patterns at different scales. The population totals dataset
        provides an overview of the overall population count for specific
        regions, while the disaggregated dataset offers a more nuanced
        understanding of demographic characteristics such as age and
        gender.<br><br>
        WorldPop offers population data at varying resolutions, including 1km
        and 100m. The tool utilizes the 100m data, as it provides a higher
        level of accuracy when estimating population for areas of interest,
        regardless of their size.<br><br>
        WorldPop's population data includes different data types. The
        <b>unconstrained</b> dataset provides estimates based on various data
        sources and modeling techniques, while the constrained dataset
        incorporates additional constraints, such as known total population or
        administrative data, to improve the accuracy of the estimates.
        Furthermore, WorldPop provides UN adjusted datasets, which align with
        the United Nations' demographic estimates and projections.<br><br>
        The availability of data varies across different years. The total
        population data spans from the year 2000 to 2020, enabling researchers
        to analyze population trends and changes over a significant time
        period. On the other hand, the disaggregated data, which provides more
        detailed information, is currently limited to the year 2020. Worldpop
        has recently released more up-to-date data for specific countries.
        However, this tool does not support retrieving data beyond 2020.
        The limitation arises from the fact that the uploaded areas of interest
        may encompass multiple countries. To maintain consistent results, it
        is necessary to have complete global coverage for a specific year.
        Unfortunately, such coverage is unavailable for years after 2020.
        <br><br>
        For more information, visit <a href='%s'>their website</a>.
    </p>
    <h3>WorldPop and Google Earth Engine</h3>
    <p align="justify">
        WorldPop's population data is accessible through their API, with the
        option to download TIF files per country. However, a challenge arises
        when dealing with areas of interest that span multiple countries or
        represent only small proportions of those countries. Downloading TIF
        files for entire countries can be inconvenient and time-consuming,
        potentially causing memory issues.<br><br>
        Thankfully, alternative methods for data retrieval exist. Google Earth
        Engine (GEE) offers <a href='%s'>access</a> to the most commonly used
        WorldPop datasets. With GEE, it becomes possible to create a mosaic of
        population data, clip the raster according to the specific areas of
        interest, and then calculate zonal statistics. This approach allows for
        a more targeted extraction of population information without the need
        to download complete country-level TIF files.<br><br>
        For most shapefiles, the tool efficiently conducts all the computations
        on the server side of Google Earth Engine (GEE). This means that the
        heavy computational tasks, such as processing and analyzing the
        population data, are performed within the GEE infrastructure. As a
        result, users can avoid the need to download the entire dataset and
        instead retrieve only the final population figures, optimizing both
        time and storage resources.<br><br>
        Unfortunately, GEE has limitations for the size of the data that can
        be uploaded (usually restricted to 10 MB). To overcome such
        limitations, a workaround is required. The solution involves creating
        a bounding box that encompasses the geometries of interest, importing
        it into GEE, clipping the raster data within GEE, downloading the
        resulting TIF file <a href='%s'>using <i>geemap</i></a>, and performing
        the calculation of zonal statistics locally.
    </p>
    <h3>Zonal statistics</h3>
    <p align="justify">
        When all computations are performed within GEE, the calculation of
        zonal statistics is seamlessly integrated into the GEE workflow. GEE
        provides built-in methods and functions specifically designed for zonal
        statistics, allowing users to extract statistical information, such as
        mean, sum, or maximum, directly within the GEE environment. In
        cases where large shapefiles are involved, the zonal statistics
        computation are performed locally. The library <i>rasterio</i>
        offers methods that enable the
        calculation of zonal statistics outside of the GEE environment.
    </p>
    """
    % (
        config["url_worldpop"],
        config["url_worldpop_GEE"],
        config["url_geemap_download"],
    ),
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h3>Summary</h3>
    <p align="justify">
        The images below illustrate the two methodologies described above,
        one for small shapefiles with simple geometries and the other for
        large shapefiles with complex geometries. When a step is labeled as
        "GEE," it signifies that the process is carried out within GEE.
        Conversely, the label "Local" indicates that the process is performed
        on the client side, locally. The second methodology requires
        significantly more computational time since multiple TIFF files must
        be downloaded prior to calculating zonal statistics. The whole process
        needs to be repeated for each requested population figure. When
        querying aggregated data, only one raster is retrieved from the server,
        whereas disaggregated data requires the processing of 36 rasters (18
        age groups for 2 genders).
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h4>1. Small shapefiles</h4>",
    unsafe_allow_html=True,
)
image = Image.open("app/img/method_easy.png")
st.image(image, use_column_width="always")

st.markdown(
    "<h4>2. Large shapefiles</h6>",
    unsafe_allow_html=True,
)
image = Image.open("app/img/method_complex.png")
st.image(image, use_column_width="always")

st.markdown(
    "<br><br>",
    unsafe_allow_html=True,
)

st.markdown("""---""")
# Limitation section
st.markdown("## Key limitations")
st.markdown(
    """
    <h3>Worldpop data</h3>
    <ul>
        <li><p align="justify">
            Spatial and temportal resolution may vary, depending on the region
            and data source. In some cases, the data may not capture very
            small-scale population variations or recent changes due to the lag
            in data availability or limitations in the input data sources.
            To maintain simplicity and significance, population figures are
            rounded to the nearest hundred.
        </p>
        <li><p align="justify">
            The accuracy of WorldPop's population estimates may be influenced
            by the inherent uncertainty associated with the modeling and
            statistical techniques used for estimation.
        </p>
        <li><p align="justify">
            Data availability may differ across regions and years, with some
            areas having more comprehensive data coverage than others.
        </p>
        <li><p align="justify">
            The disaggregated data is currently limited to the year 2020,
            restricting the ability to analyze population dynamics over an
            extended time period.
        </p>
    </ul>
    <h3>Zonal statistics algorithms</h3>
    <ul>
        <li><p align="justify">
            The algorithms may have limitations in handling
            irregular or complex polygon geometries, leading to potential
            inaccuracies in the calculated statistics.
        </p>
        <li><p align="justify">
            The algorithms may struggle with edge effects, where pixel values
            near the boundary of a zone may be influenced by neighboring zones,
            affecting the accuracy of the statistics.
        </p>
        <li><p align="justify">
            If the resolution of the raster data is significantly different
            from the polygon boundaries, errors or biases may occur due to
            pixel aggregation or disaggregation.
        </p>
    </ul>
    """,
    unsafe_allow_html=True,
)

# Link section
# st.markdown("## Useful links")
