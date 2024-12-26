import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from joblib import load
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from components.about import show_contact_information
high_res_columns = [
    "POINT_X", "POINT_Y", "GLDAS_SerialID", "TWI", "TRI", "Sy", "STI", "SPI", "Slope",
    "Profile_curvature", "Plan_curvature", "lithology", "lithology_clay_thickness",
    "Distance_from_stream", "elevation", "drainage_density", "Curvature", "Aspect",
    "NDVI", "NDWI"
]

low_res_columns = [
    "POINT_X", "POINT_Y", "GLDAS_SerialID", "Curvature", "Slope_o_", "Profile_curvature",
    "Plan_curvature", "Distance_from_stream", "Aspect", "drainage_density_MAJORITY",
    "Elevation_MAJORITY", "lithology_MAJORITY", "SPI_MAJORITY", "STI_MAJORITY",
    "Sy_MAJORITY", "TRI_MAJORITY", "TWI_MAJORITY", "lithology_clay_thickness_MAJORITY",
    "NDVI", "NDWI", "Min_GWS", "Max_GWS"
]

def upload_and_process_page():
    # Load necessary models and scalers
    standard_scaler = load('model weights/standard_scaler.pkl')
    fitted_scaler = load('model weights/fitted_scaler.pkl')
    rf_model = load('model weights/upsampling_model.joblib')

    # Page title
    st.write("GLDAS Groundwater Level Downscaling App. Upsampling from low resolution (25 km) GLDAS to high resolution (2 km) in-situ measurements.")
    
    # In-memory storage for uploaded data
    high_res_data = None
    low_res_data = None
    


    # Utility function to create scatter plots with discrete colors
    def plot_discrete_color_map(data, column, color_title):
        boundaries = [0, 5.3, 7.6, 9.8, 11.3, 15, 20.5, 26, 35.5, 58, 60, 70, 80, 90, 100, 150]
        cmap_colors = [
            'blue', '#a8e1e0', '#66c18a', '#3b7a3d', '#f3d5a4', '#b299ca',
            '#e4a6a3', '#d35d60', '#a0322e', '#330e0f', '#4f4d4d', '#7d7b7b',
            '#a9a8a8', '#c2c0c0', '#dbdbdb', 'black'
        ]

        def get_discrete_color(val):
            for i in range(len(boundaries) - 1):
                if boundaries[i] <= val < boundaries[i + 1]:
                    return cmap_colors[i]
            return cmap_colors[-1]

        data[f"{column}_color"] = data[column].apply(get_discrete_color)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=data['POINT_X'],
            y=data['POINT_Y'],
            hue=data[f"{column}_color"],
            palette=cmap_colors,
            legend=None
        )
        plt.title(color_title)
        plt.xlabel("POINT_X")
        plt.ylabel("POINT_Y")
        st.pyplot(plt)

    # Sidebar for uploading files
    st.header("Upload Data")
    high_col,low_col=st.columns(2)
    with high_col:
        high_res_file = st.file_uploader("Upload High-Resolution data from ArcGIS CSV", type=["csv"], key="high_res")
    with low_col:
        low_res_file = st.file_uploader("Upload Low-Resolution data from GLDAS CSV", type=["csv"], key="low_res")

    if high_res_file:
        high_res_data = pd.read_csv(high_res_file)
        missing_high_res_columns = set(high_res_columns) - set(high_res_data.columns)
        if missing_high_res_columns:
            st.error(f"High-Resolution data is missing columns: {missing_high_res_columns}")
            high_res_data=None
        elif high_res_data.isna().any().any():
            st.error("High-Resolution dataset contains missing values. Please clean the data and re-upload.")
            high_res_data=None
        else:
            high_res_data['original_Sy'] = high_res_data['Sy']
            high_res_data[fitted_scaler.feature_names_in_] = fitted_scaler.transform(
                high_res_data[fitted_scaler.feature_names_in_]
            )
            st.sidebar.success("High-Resolution Data uploaded and scaled successfully!")

    if low_res_file:
        low_res_data = pd.read_csv(low_res_file)
        missing_low_res_columns = set(low_res_columns) - set(low_res_data.columns)
        if missing_low_res_columns:
            st.error(f"Low-Resolution data is missing columns: {missing_low_res_columns}")
            low_res_data=None
        elif low_res_data.isna().any().any():
            st.error("Low-Resolution dataset contains missing values. Please clean the data and re-upload.")
            low_res_data=None
        else:
            st.sidebar.success("Low-Resolution Data uploaded successfully!")

    # Process data if both files are uploaded
    if high_res_data is not None and low_res_data is not None:
        st.header("Data Processing")

        # Merge high- and low-resolution data
        try:
            merged_data = high_res_data.merge(low_res_data, on=['GLDAS_SerialID'])
            merged_data.rename(columns={'POINT_X_x': 'POINT_X', 'POINT_Y_x': 'POINT_Y'}, inplace=True)
            if merged_data.isnull().any().any():
                st.error("'GLDAS_SerialID' not matching. Please make sure both datasets have same 'GLDAS_SerialID'.")
                return
        except KeyError:
            st.error("Both datasets must have 'GLDAS_SerialID' and 'Year' columns for merging.")
            return

        # Check for missing values in numerical columns
        numerical_columns = merged_data.select_dtypes(include=['number']).columns
        if merged_data[numerical_columns].isna().any().any():
            st.error("Numerical columns contain missing (NaN) values. Please clean the data and re-upload.")
            return

        # Check for invalid categories in categorical columns
        categorical_columns = ['lithology', 'lithology_MAJORITY']
        for col in categorical_columns:
            if col not in merged_data.columns:
                st.error(f"Missing required categorical column: {col}")
                return

        # Scale numerical features
        X_numerical_scaled = standard_scaler.transform(merged_data[standard_scaler.feature_names_in_])

        # Encode categorical data
        X_cat = merged_data[categorical_columns].astype('string')
        X_cat_encoded = pd.get_dummies(X_cat)

        # Prepare final input tensor
        X_final = torch.tensor(np.hstack([X_numerical_scaled, X_cat_encoded]).astype('float32'))

        # Predict
        pred = rf_model.predict(X_final)
        recharge = (pred[:, 1] - pred[:, 0]) * 100 * merged_data['original_Sy']
        merged_data['Min_GWL'] = pred[:, 0]
        merged_data['Max_GWL'] = pred[:, 1]
        merged_data['Recharge'] = recharge

        # Display merged data
        st.subheader("Upsampled Data")
        upsampled = merged_data[['POINT_X', 'POINT_Y', 'Min_GWL', 'Max_GWL', 'Recharge']]
        st.write(upsampled.head())

        # Visualization options
        st.subheader("Visualization")
        selected_plot = st.selectbox("Select Plot Type", ["Max GWL", "Min GWL", "Recharge"])

        if selected_plot == "Max GWL":
            plot_discrete_color_map(merged_data, 'Max_GWL', "Max Groundwater Level")
        elif selected_plot == "Min GWL":
            plot_discrete_color_map(merged_data, 'Min_GWL', "Min Groundwater Level")
        else:
            plot_discrete_color_map(merged_data, 'Recharge', "Recharge")

        # Provide download option
        csv = upsampled.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="upsampled_data.csv">Download Upsampled Data</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("Please upload both High-Resolution and Low-Resolution data to proceed. Please follow the requirements.")
    if high_res_data is None or low_res_data is None:
        # Display dataset requirements
        st.header("Dataset Requirements")
        # Streamlit app for downloading test data
        st.write("Download the test data to explore the dataset for a specific year in Bangladesh. This data will help you understand the input format for both high-resolution and low-resolution CSV files.")
        with open("test_data.zip", "rb") as file:
            st.download_button(
                label="Download Test Data", 
                data=file, 
                file_name="test_data.zip", 
                mime="application/zip"
            )

        st.markdown("### General Requirements")
        st.markdown("""
        - Both High-Resolution and Low-Resolution datasets must include the following columns:
        - **`GLDAS_SerialID`**: For merging.
        - **`Year`**: For temporal alignment.
        - No missing (NaN) values in numerical columns.
        - Categorical columns must only contain valid categories.
        """)

        st.markdown("### High-Resolution Dataset")
        st.markdown("""
        The High-Resolution dataset must include the following columns:
        - **Geographical Information for plotting:**
        - `POINT_X` (longitude)
        - `POINT_Y` (latitude)
        - **Hydrological and Terrain Features:**
        - `TWI`, `TRI`, `Sy`, `STI`, `SPI`, `Slope`
        - `Profile_curvature`, `Plan_curvature`, `Curvature`, `Aspect`
        - **Land Cover and Lithology:**
        - `lithology`, `lithology_clay_thickness`
        - `Distance_from_stream`, `elevation`, `drainage_density`
        - **Vegetation Indices:**
        - `NDVI`, `NDWI`
       
        """)

        st.markdown("### Low-Resolution Dataset")
        st.markdown("""
        The Low-Resolution dataset must include the following columns:
        - **Geographical Information for plotting:**
        - `POINT_X` (longitude)
        - `POINT_Y` (latitude)
        - **Hydrological and Terrain Features:**
        - `Curvature`, `Slope_o_`, `Profile_curvature`, `Plan_curvature`
        - `Distance_from_stream`, `Aspect`
        - `drainage_density_MAJORITY`, `Elevation_MAJORITY`
        - **Lithology and Soil Properties:**
        - `lithology_MAJORITY`, `SPI_MAJORITY`, `STI_MAJORITY`
        - `Sy_MAJORITY`, `TRI_MAJORITY`, `TWI_MAJORITY`
        - `lithology_clay_thickness_MAJORITY`
        - **Vegetation Indices:**
        - `NDVI`, `NDWI`
        - **Groundwater Levels:**
        - `Min_GWS`, `Max_GWS`
        
        """)

        st.markdown("### Key Notes:")
        st.markdown("""
        - `POINT_X` represents longitude and `POINT_Y` represents latitude.
        - Categorical Columns:
        - High-Resolution: `lithology`
        - Low-Resolution: `lithology_MAJORITY`
        - Ensure all numerical columns are free of missing values.
        - Avoid invalid or unexpected categories in categorical columns.
        - Give data for only one year.
        """)
        st.warning("Due to constrain on the deployed environment we have used a very light version of the model. Original model is given in the notebooks in our paper. If you want to support us to create a more powerful website please contact us.")
        # st.page_link(st.Page(show_contact_information,title="Contact us"))

