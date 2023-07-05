"""Module for ee-related functionalities."""
import ee
import streamlit as st
from ee import oauth
from google.oauth2 import service_account


@st.cache_data(show_spinner="Initializing Google Earth Engine...")
def ee_initialize():
    """Initialise Google Earth Engine."""
    service_account_keys = st.secrets["ee_keys"]
    credentials = service_account.Credentials.from_service_account_info(
        service_account_keys, scopes=oauth.SCOPES
    )
    ee.Initialize(credentials)
