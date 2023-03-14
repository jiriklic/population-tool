"""Functions for population data."""
import os
import urllib.request
from ftplib import FTP
from typing import Any, Dict, List, Optional, Tuple, Union

import country_converter as coco
import geopandas as gpd
import numpy as np
import rasterio
import requests
import shapely
import streamlit as st
from rasterio.features import shapes
from rasterstats import zonal_stats
from shapely.geometry import shape
from shapely.ops import unary_union


def mock_st_text(text: str, verbose: bool, text_on_streamlit: bool) -> None:
    """
    Return streamlit text using a given text object.

    If the object is None, return None. This function can be used in streamlit
    or in jupyter notebook.

    Inputs
    ----------
    text (str): text to display.
    verbose (bool): if True, print details run.
    text_on_streamlit (str): if True, print text in the Streamlit app
        via st.write(), otherwise print to console.

    Returns
    -------
    None or the updated text_field.
    """
    if not verbose:
        return
    if text_on_streamlit:
        st.write(text)
    else:
        print(text)


def check_gdf_geometry(gdf: gpd.GeoDataFrame) -> bool:
    """Check whether all geometries are polygons or multipolygons.

    Inputs
    ----------
    pgdf (geopandas.GeoDataFrame): GeoDataFrame with polygons.

    Returns
    -------
    (bool): True if all geometries are points, False otherwise.
    """
    return all(["Polygon" in geom for geom in gdf.geom_type.tolist()])


def find_intersections_polygon(
    pol: shapely.Polygon,
    world_gdf: gpd.GeoDataFrame,
) -> Dict:
    """
    Find the intersections between a Polygon object and country borders.

    Inputs
    ----------
    pol (shapely.Polygon): shapely Polygon object.
    world_gdf (geopandas.GeoDataFrame): Geodataframe containing polygons of
        world countries.

    Returns
    -------
    intersect_dict (dict): dictionary with
        - keys containing ISO3 country codes corresponding to the countries
        intersected by the input Polygon,
        - values containing intersected portions of each country's polygon, as
        Polygon objects.
    """
    ind_intersect = pol.intersects(world_gdf["geometry"])
    world_intersect = world_gdf[
        world_gdf.index.isin(ind_intersect[ind_intersect].index.values)
    ]

    return dict(
        [
            (
                world_intersect["iso3"].iloc[i],
                world_intersect["geometry"].iloc[i].intersection(pol),
            )
            for i in range(len(world_intersect))
        ]
    )


def find_intersections_gdf(
    gdf: gpd.GeoDataFrame,
    data_folder: str = "app/data",
    tif_folder: str = "app/test_data/pop_data",
) -> Tuple[Dict, List[str]]:
    """
    Find the intersections between polygons in a dataframe and country borders.

    Country borders are retrieved from WorldPop (1km resolution).

    Inputs
    ----------
    gdf (geopandas.GeoDataFrame): GeoDataFrame containing Polygon geometries.
    data_folder (str, optional): folder where auxiliary data was saved (such
        as country borders).
    tif_folder (str, optional): folder where the world borders raster data is
        saved.

    Returns
    -------
    intersect_dict (dict): dictionary where the keys are the indices of the
        input GeoDataFrame and the values are dictionaries of intersected
        country polygons, keyed by ISO3 country codes.
    all_iso3_list (list): list of all ISO3 country codes intersected by any of
        the input polygons.
    """
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    filename = f"{data_folder}/world_borders_wp.geojson"

    if os.path.isfile(filename):
        world_gdf = gpd.read_file(filename)
    else:
        world_gdf = retrieve_country_borders_wp(tif_folder)

    intersect_list = [
        find_intersections_polygon(geom, world_gdf=world_gdf)
        for geom in gdf["geometry"].tolist()
    ]
    intersect_dict = dict(zip(range(len(intersect_list)), intersect_list))
    all_iso3_list = list(
        set().union(*[list(inter.keys()) for inter in intersect_list])
    )  # type: list

    return intersect_dict, all_iso3_list


def retrieve_country_borders_wp(
    tif_folder: str = "app/test_data/pop_data",
) -> gpd.GeoDataFrame:
    """
    Retrieve country borders from WorldPop.

    Country borders are retrieved from WorldPop (1km resolution) in raster
    format. They are then reformatted into vectors and attached to a
    geopandas.GeoDataFrame. The resulting dataframe has iso3 and geometries as
    columns.

    Inputs
    ----------
    tif_folder (str, optional): folder where the world borders raster data is
        saved.

    Returns
    -------
    world_gdf (geopandas.GeoDataFrame): GeoDataFrame containing country borders
        of the world, according to WorldPop.
    """
    tif_url = (
        "https://data.worldpop.org/GIS/Mastergrid/Global_2000_2020/0_Mosaicked"
        "/global_mosaic_1km/global_level0_1km_2000_2020.tif"
    )

    raster_file = f'{tif_folder}/{tif_url.split("/")[-1]}'
    urllib.request.urlretrieve(tif_url, raster_file)

    with rasterio.open(raster_file) as src:
        data = src.read(1, masked=True)

        # Use a generator instead of a list
        shape_gen = (
            (shape(s), v) for s, v in shapes(data, transform=src.transform)
        )
        gdf = gpd.GeoDataFrame(
            dict(zip(["geometry", "value"], zip(*shape_gen))), crs=src.crs
        )

    cc = coco.CountryConverter()

    unique_iso_num_list = list(
        set([int(iso) for iso in gdf["value"].tolist() if iso != 65535])
    )
    unique_iso_str_list = cc.convert(names=unique_iso_num_list, to="ISO3")

    iso3_list = []
    geometry_list = []
    for i, (iso_num, iso_str) in enumerate(
        zip(unique_iso_num_list, unique_iso_str_list)
    ):
        pol = unary_union(gdf[gdf["value"] == iso_num]["geometry"].tolist())
        iso3_list.append(iso_str)
        geometry_list.append(pol)

    world_gdf = gpd.GeoDataFrame(
        data={"iso3": iso3_list}, geometry=geometry_list, crs=gdf.crs
    )

    world_gdf.iat[
        world_gdf[world_gdf["iso3"] == "not found"].iloc[[0]].index[0], 0
    ] = "KOS"
    return (
        world_gdf[world_gdf["iso3"] != "not found"]
        .sort_values(by="iso3")
        .reset_index(drop=True)
    )


def download_worldpop_iso_tif(
    iso3: str,
    tif_folder: str,
    method: str = "http",
    year: int = 2020,
    data_type: str = "UNadj_constrained",
    aggregated: bool = True,
    clobber: bool = False,
    progress_bar: bool = False,
    step_progression: float = 0.0,
    step_size: float = 0.0,
    my_bar: Optional[Any] = None,
    progress_text: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Download WorldPop raster data for given parameters.

    Inputs:
    -------
    iso3 (str): ISO3 country code.
    tif_folder (str): folder name, where download raster data are saved.
    method (str, optional): download method (http API or ftp service).
        ['http' | 'ftp']. Default to 'http'.
    year (int, optional): year of population data. Default to 2020.
    data_type (str, optional): type of population estimate.
        ['unconstrained'| 'constrained' | 'UNadj_constrained']. Default to
        'UNadj_constrained'.
    aggregated (bool, optional): if False, download disaggregated data (by
        gender and age), if True download only total population figure.
        Default to True.
    clobber (bool, optional): if True, overwrite data, if False do not
        download data if already present in the folder. Default to False.
    progress_bar (bool, optional): if True, create a progress bar in Streamlit.
    step_progression (float, optional): number that indicates the progression
        of the computation. Values between 0 and 1. Default to 0.
    step_size (float, optional): number that indicates the step size for the
        progress bar. Values between 0 and 1. Default to 0.
    mybar (st.progress, optional): progress bar object. Default to None.
    progress_text (str, optional): text to visualize above the progress bar.
        Default to None.

    Returns:
    --------
    filename_list (list): list of filenames of the GeoTIFF files containing the
        population data for the given parameters.
    label_list (list): list of labels for dataframe columns. If the population
        data is disaggregated, each label indicates gender and age.
    """
    allowed_method = ["http", "ftp"]
    if method not in allowed_method:
        raise ValueError(
            "method should be one of the options: " + ", ".join(allowed_method)
        )

    filename_base = f"{tif_folder}/{iso3.lower()}_{year}_{data_type}"

    if method == "http":
        return download_worldpop_iso3_tif_http(
            iso3,
            year=year,
            data_type=data_type,
            aggregated=aggregated,
            filename_base=filename_base,
            clobber=clobber,
            progress_bar=progress_bar,
            step_progression=step_progression,
            step_size=step_size,
            my_bar=my_bar,
            progress_text=progress_text,
        )
    else:
        return download_worldpop_iso3_tif_ftp(
            iso3,
            year=year,
            data_type=data_type,
            aggregated=aggregated,
            filename_base=filename_base,
            clobber=clobber,
            progress_bar=progress_bar,
            step_progression=step_progression,
            step_size=step_size,
            my_bar=my_bar,
            progress_text=progress_text,
        )


def download_worldpop_iso3_tif_http(
    iso3: str,
    data_type: str = "UNadj_constrained",
    aggregated: bool = True,
    filename_base: Optional[str] = None,
    year: int = 2020,
    clobber: bool = False,
    progress_bar: bool = False,
    step_progression: float = 0.0,
    step_size: float = 0.0,
    my_bar: Any = st.progress(0),
    progress_text: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Download WorldPop raster data for given parameters, using the API via http.

    Inputs:
    -------
    iso3 (str): ISO3 country code.
    data_type (str, optional): type of population estimate.
        ['unconstrained'| 'constrained' | 'UNadj_constrained']. Default to
        'UNadj_constrained'.
    aggregated (bool, optional): if False, download disaggregated data (by
        gender and age), if True download only total population figure.
        Default to True.
    filename_base (str, optional): part of the filename of the downloaded
        raster file.
    year (int, optional): year of population data. Default to 2020.
    clobber (bool, optional): if True, overwrite data, if False do not
        download data if already present in the folder. Default to False.
    progress_bar (bool, optional): if True, create a progress bar in Streamlit.
    step_progression (float, optional): number that indicates the progression
        of the computation. Values between 0 and 1. Default to 0.
    step_size (float, optional): number that indicates the step size for the
        progress bar. Values between 0 and 1. Default to 0.
    mybar (st.progress, optional): progress bar object. Default to None.
    progress_text (str, optional): text to visualize above the progress bar.
        Default to None.

    Returns:
    --------
    filename_list (list): list of filenames of the GeoTIFF files containing the
        population data for the given parameters.
    label_list (list): list of labels for dataframe columns. If the population
        data is disaggregated, each label indicates gender and age.
    """
    if aggregated:
        api_suffix_1 = "age_structures/"
    else:
        api_suffix_1 = "pop/"

    if data_type == "unconstrained" and aggregated:
        api_suffix_2 = "wpgp"
    elif data_type == "unconstrained" and not aggregated:
        api_suffix_2 = "aswpgp"
    else:
        if year != 2020:
            raise ValueError("Constrained data is only available for 2020.")
        else:
            if data_type == "constrained" and aggregated:
                api_suffix_2 = f"cic{year}_100m"
            elif data_type == "constrained" and not aggregated:
                api_suffix_2 = f"ascic_{year}"
            elif data_type == "UNadj_constrained" and aggregated:
                api_suffix_2 = f"cic{year}_UNadj_100m"
            elif data_type == "UNadj_constrained" and not aggregated:
                api_suffix_2 = f"ascicua_{year}"

    api_url = (
        f"https://hub.worldpop.org/rest/data/{api_suffix_1}{api_suffix_2}"
        f"?iso3={iso3}"
    )

    response = requests.get(api_url)
    data_json = response.json()["data"]

    if data_type == "unconstrained":
        year_index = [
            i
            for (i, data_year) in enumerate(data_json)
            if int(data_year["popyear"]) == year
        ][0]
    else:
        year_index = 0

    tif_url_list = sorted(data_json[year_index]["files"])

    filename_list = []
    label_list = []

    for j, tif_url in enumerate(tif_url_list):
        if filename_base is None:
            filename = tif_url.split("/")[-1]
            label = f"pop_{j}"
        else:
            if aggregated:
                filename = f"{filename_base}.tif"
                label = "pop_tot"
            else:
                gender = tif_url.split("/")[-1].split("_")[1]
                age = tif_url.split("/")[-1].split("_")[2].zfill(2)
                filename = f"{filename_base}_{gender}_{age}.tif"
                label = f"pop_{gender}_{age}"

        if clobber:
            condition_to_download = True
        else:
            condition_to_download = not os.path.isfile(filename)

        if condition_to_download:
            urllib.request.urlretrieve(tif_url, filename)

        filename_list.append(filename)
        label_list.append(label)

        if progress_bar:
            step_progression += step_size
            my_bar.progress(step_progression, text=progress_text)

    filename_list = [
        filename for _, filename in sorted(zip(label_list, filename_list))
    ]
    label_list.sort()

    return filename_list, label_list


def download_worldpop_iso3_tif_ftp(
    iso3: str,
    data_type: str = "UNadj_constrained",
    aggregated: bool = True,
    filename_base: Optional[str] = None,
    year: int = 2020,
    clobber: bool = False,
    progress_bar: bool = False,
    step_progression: float = 0.0,
    step_size: float = 0.0,
    my_bar: Any = st.progress(0),
    progress_text: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Download WorldPop raster data for given parameters, using the ftp service.

    Inputs:
    -------
    iso3 (str): ISO3 country code.
    data_type (str, optional): type of population estimate.
        ['unconstrained'| 'constrained' | 'UNadj_constrained']. Default to
        'UNadj_constrained'.
    aggregated (bool, optional): if False, download disaggregated data (by
        gender and age), if True download only total population figure.
        Default to True.
    filename_base (str, optional): part of the filename of the downloaded
        raster file.
    year (int, optional): year of population data. Default to 2020.
    clobber (bool, optional): if True, overwrite data, if False do not
        download data if already present in the folder. Default to False.
    progress_bar (bool, optional): if True, create a progress bar in Streamlit.
    step_progression (float, optional): number that indicates the progression
        of the computation. Values between 0 and 1. Default to 0.
    step_size (float, optional): number that indicates the step size for the
        progress bar. Values between 0 and 1. Default to 0.
    mybar (st.progress, optional): progress bar object. Default to None.
    progress_text (str, optional): text to visualize above the progress bar.
        Default to None.

    Returns:
    --------
    filename_list (list): list of filenames of the GeoTIFF files containing the
        population data for the given parameters.
    label_list (list): list of labels for dataframe columns. If the population
        data is disaggregated, each label indicates gender and age.
    """
    ftp = FTP("ftp.worldpop.org.uk")
    ftp.login()

    if aggregated:
        folder_base = "/GIS/Population/Global_2000_2020"
    else:
        folder_base = "/GIS/AgeSex_structures/Global_2000_2020"

    if data_type == "unconstrained":
        file_suffix = ""
        folder = f"{folder_base}/{year}/{iso3}/"
        ftp.cwd(folder)
        file_list = ftp.nlst()
        if aggregated:
            file_list = [file_list[0]]
    else:
        if year != 2020:
            raise ValueError("Constrained data is only available for 2020.")
        if aggregated:
            file_suffix = f"_{data_type}"
            folder_base = f"{folder_base}_Constrained/{year}/"

            ftp_sources = ["maxar_v1/", "BSGM/"]
            for source in ftp_sources:
                ftp.cwd(f"{folder_base}/{source}")
                if iso3 in ftp.nlst():
                    break
            folder = f"{folder_base}/{source}{iso3}/"
            ftp.cwd(folder)
            file_list = [f"{iso3.lower()}_ppp_{year}{file_suffix}.tif"]
        else:
            if data_type == "constrained":
                folder_base = f"{folder_base}_Constrained"
            else:
                folder_base = f"{folder_base}_Constrained_UNadj"
            folder = f"{folder_base}/{year}/{iso3}/"
            ftp.cwd(folder)
            file_list = ftp.nlst()

    filename_list = []
    label_list = []

    for j, file in enumerate(file_list):
        if filename_base is None:
            filename = file
            label = f"pop_{j}"
        else:
            if aggregated:
                filename = f"{filename_base}.tif"
                label = "pop_tot"
            else:
                gender = file.split("_")[1]
                age = file.split("_")[2].zfill(2)
                filename = f"{filename_base}_{gender}_{age}.tif"
                label = f"pop_{gender}_{age}"

        if clobber:
            condition_to_download = True
        else:
            condition_to_download = not os.path.isfile(filename)

        if condition_to_download:
            with open(filename, "wb") as file_out:
                ftp.retrbinary(f"RETR {file}", file_out.write)

        filename_list.append(filename)
        label_list.append(label)

        if progress_bar:
            step_progression += step_size
            my_bar.progress(step_progression, text=progress_text)

    filename_list = [
        filename for _, filename in sorted(zip(label_list, filename_list))
    ]
    label_list.sort()

    ftp.quit()

    return filename_list, label_list


def aggregate_raster_on_geometries(
    raster_file: str,
    raster_dict: dict,
    geometry_list: List[shapely.Geometry],
    stats: Union[str, List[str]] = "sum",
) -> Tuple[List, dict]:
    """
    Compute zonal statistics of a raster file over a list of vector geometries.

    Inputs:
    -------
    raster_file (str): filepath or url to the input raster file.
    raster_dict (dict): dictionary containing raster filepaths as keys and
        arrays and affine transformed rasters as values. If raster_file is
        already listed in raster_dict, there is no need to recalculate array
        and affine transformed raster.
    geometry_list (list): list of shapely geometries (e.g. polygons) over which
        to compute the zonal statistics.
    stats (str or list, optional): One or more statistics to compute for each
        geometry, such as 'mean', 'sum', 'min', 'max', 'std', etc. Default to
        'sum'.

    Returns:
    --------
    stats_list (list): List of zonal statistics computed for each input
        geometry, in the same order as the input list. Each item in the list is
        a dictionary containing the zonal statistics for a single geometry,
        with keys like 'sum', 'mean', etc. depending on the input 'stats'
        parameter.
    raster_dict (dict): new raster_dict, which contains data about raster_file.
    """
    if raster_file in raster_dict.keys():
        array = raster_dict[raster_file]["array"]
        affine = raster_dict[raster_file]["affine"]
    else:
        raster = rasterio.open(raster_file, nodata=0.0)
        affine = raster.transform
        array = raster.read(1)
        array = array.astype(float)
        array[array < 0] = np.nan
        raster_dict[raster_file] = {"array": array, "affine": affine}

    return (
        zonal_stats(geometry_list, array, affine=affine, stats=stats),
        raster_dict,
    )


def add_population_data(
    gdf: gpd.GeoDataFrame,
    data_type: str = "UNadj_constrained",
    year: int = 2020,
    aggregated: bool = True,
    data_folder: str = "app/data",
    tif_folder: str = "app/test_data/pop_data",
    clobber: bool = False,
    verbose: bool = False,
    text_on_streamlit: bool = True,
    progress_bar: bool = False,
) -> gpd.GeoDataFrame:
    """
    Add population data to a GeoDataFrame by aggregating WorldPop data.

    Since WorldPop population data can only be downloaded per country, the
    function calculates the intersections between the geometries in the
    dataframe and each one of the world countries. Then it calculates zonal
    statistics for each pair (geometry, country) and aggregate the results
    per geometry.

    Parameters
    ----------
    gdf (geopandas.GeoDataFram: GeoDataFrame containing the geometries for
        which to add population data.
    data_type (str, optional): type of population estimate.
        ['unconstrained'| 'constrained' | 'UNadj_constrained']. Default to
        'UNadj_constrained'.
    year (int, optional): year of population data. Default to 2020.
    aggregated (bool, optional): if False, download disaggregated data (by
        gender and age), if True download only total population figure.
        Default to True.
    data_folder (str, optional): folder where auxiliary data was saved (such
        as country borders).
    tif_folder (str, optional): folder where the population raster data is
        saved.
    clobber (bool, optional): if True, overwrite data, if False do not
        download data if already present in the folder. Default to False.
    verbose (bool, optional): if True, print details run. Default to False.
    text_on_streamlit (bool, optional): if True, print to streamlit, otherwise
        via the print statement. Only used when verbose is True.
        Default to True.
    progress_bar (bool, optional): if True, create a progress bar in Streamlit.

    Returns
    -------
    gdf_with_pop (geopandas.GeoDataFrame): input GeoDataFrame with an
        additional 'total population' column containing the aggregated
        population data.
    """
    if progress_bar:
        progress_text = "Operation in progress. Please wait."
    else:
        progress_text = ""
    my_bar = st.progress(0, text=progress_text)

    gdf_with_pop = gdf.copy()

    allowed_data_type = ["unconstrained", "constrained", "UNadj_constrained"]
    if data_type not in allowed_data_type:
        raise ValueError(
            "data_type should be one of the options: "
            + ", ".join(allowed_data_type)
        )

    mock_st_text(
        verbose=verbose,
        text_on_streamlit=text_on_streamlit,
        text="Finding intersections with countries...",
    )
    pop_total_dict = {}  # type: dict
    intersect_dict, all_iso3_list = find_intersections_gdf(
        gdf, data_folder=data_folder, tif_folder=tif_folder
    )
    intersect_countries_str = ", ".join(all_iso3_list)
    mock_st_text(
        verbose=verbose,
        text_on_streamlit=text_on_streamlit,
        text=f"Intersected countries: {intersect_countries_str}.",
    )

    if not os.path.exists(tif_folder):
        os.makedirs(tif_folder)

    if progress_bar:
        if aggregated:
            number_of_steps = len(all_iso3_list)
        else:
            number_of_steps = len(all_iso3_list) * 36

        step_size = 0.8 / number_of_steps
        step_progression = 0.1
        my_bar.progress(step_progression, text=progress_text)
    else:
        step_progression = 0.0
        step_size = 0.0

    raster_dict = {}  # type: dict

    for i in range(len(all_iso3_list)):
        iso3 = all_iso3_list[i]
        mock_st_text(
            verbose=verbose,
            text_on_streamlit=text_on_streamlit,
            text=f"Country: {iso3}.",
        )
        iso3_list_indexes = [
            i for i, value in intersect_dict.items() if iso3 in value.keys()
        ]
        iso3_list_geometries = [
            value[iso3]
            for i, value in intersect_dict.items()
            if iso3 in value.keys()
        ]

        mock_st_text(
            verbose=verbose,
            text_on_streamlit=text_on_streamlit,
            text="Downloading raster...",
        )
        raster_file_list, label_list = download_worldpop_iso_tif(
            iso3,
            tif_folder=tif_folder,
            data_type=data_type,
            year=year,
            aggregated=aggregated,
            clobber=clobber,
            progress_bar=progress_bar,
            step_progression=step_progression,
            step_size=step_size,
            my_bar=my_bar,
            progress_text=progress_text,
        )

        if progress_bar:
            if aggregated:
                step_progression += step_size
            else:
                step_progression += step_size * 36

        pop_partial_dict = {}

        mock_st_text(
            verbose=verbose,
            text_on_streamlit=text_on_streamlit,
            text="Aggregating raster...",
        )
        for raster_file, label in zip(raster_file_list, label_list):
            if i == 0:
                pop_total_dict[label] = {}

            pop_iso3_agg, raster_dict = aggregate_raster_on_geometries(
                raster_file=raster_file,
                geometry_list=iso3_list_geometries,
                raster_dict=raster_dict,
            )
            pop_partial_dict[label] = dict(
                zip(iso3_list_indexes, [pop["sum"] for pop in pop_iso3_agg])
            )

            for key, value in pop_partial_dict[label].items():
                if value is not None:
                    if key not in pop_total_dict[label]:
                        pop_total_dict[label][key] = value
                    else:
                        pop_total_dict[label][key] += value

            if progress_bar:
                new_step_size = 0.1 / number_of_steps
                step_progression += new_step_size

                my_bar.progress(step_progression, text=progress_text)

    for label in label_list:
        gdf_with_pop[label] = [
            int(pop_total_dict[label][i]) for i in gdf.index
        ]

    if progress_bar:
        my_bar.progress(1.0, text=" ")

    mock_st_text(
        verbose=verbose, text_on_streamlit=text_on_streamlit, text="Done!"
    )

    return gdf_with_pop
