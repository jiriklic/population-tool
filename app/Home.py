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
    This web app allows you to upload shapefiles and retrieve population data
    from WorldPop.
"""
)

# Second section
st.markdown("## How to use the tool")
