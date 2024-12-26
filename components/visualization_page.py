import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json

@st.cache_resource
def load_figures(json_dir, year):
    return {
        'max_gwl': json.load(open(f"{json_dir}/max_gwl_{year}.json")),
        'min_gwl': json.load(open(f"{json_dir}/min_gwl_{year}.json")),
        'recharge': json.load(open(f"{json_dir}/recharge_{year}.json"))
    }

@st.cache_resource
def create_cached_plot(fig_json, title):
    fig = go.Figure(fig_json)
    fig.update_traces(
        hovertemplate="Latitude (POINT_X): %{x}<br>Longitude (POINT_Y): %{y}<br>" + title + ": %{text}"
    )
    fig.update_layout(title=title)
    return fig

def visualization_page():
    # Load data
    data = pd.read_csv('pseudo_data_with_recharge.csv')
    json_dir = "figures_json"
    data = data[data.Year == 2022]
    # Set year to 2022
    year = 2022

    # Preload JSON data for the year 2022
    figures_cache = load_figures(json_dir, year)

    # Cache entire plots
    max_gwl_plot = create_cached_plot(figures_cache['max_gwl'], "Max GWL (m)")
    min_gwl_plot = create_cached_plot(figures_cache['min_gwl'], "Min GWL (m)")
    recharge_plot = create_cached_plot(figures_cache['recharge'], "Recharge (cm)")

    # Title and description
    st.header("Interactive Ground Water Level Map for the Year 2022")

    # Render plots
    # st.subheader("Interactive Plots")
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.spinner('Loading Max GWL Plot...'):
            st.plotly_chart(max_gwl_plot, use_container_width=True)

    with col2:
        with st.spinner('Loading Min GWL Plot...'):
            st.plotly_chart(min_gwl_plot, use_container_width=True)

    with col3:
        with st.spinner('Loading Recharge Plot...'):
            st.plotly_chart(recharge_plot, use_container_width=True)
