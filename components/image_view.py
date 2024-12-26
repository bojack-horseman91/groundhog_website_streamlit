import os
import streamlit as st

# Function to get image paths for a given year
def get_image_paths(year):
    base_path = "assets/figures_images"
    max_gwl_path = os.path.join(base_path, f"Upsampled_GWL/Upsample Max GWL (BGL) for the year {year} (meters).png")
    min_gwl_path = os.path.join(base_path, f"Upsampled_GWL/Upsample Min GWL (BGL) for the year {year} (meters).png")
    recharge_path = os.path.join(base_path, f"Upsampled_Recharge/Upsample Recharge for the year {year} (centimeters).png")
    return max_gwl_path, min_gwl_path, recharge_path

# Function to display the Streamlit layout for the images
def display_images():
    st.header("Yearly Ground Water Level Images")
    years=range(2003,2023)
    # Dropdown for selecting the year
    selected_year = st.selectbox(
        "Select a Year",
        years,
        index=0,
    )

    # Get image paths for the selected year
    max_gwl_path, min_gwl_path, recharge_path = get_image_paths(selected_year)

    # Display images side by side
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(f"Max GWL ({selected_year})")
        if os.path.exists(max_gwl_path):
            st.image(max_gwl_path, use_container_width=True)
        else:
            st.warning("Image not found.")

    with col2:
        st.write(f"Min GWL ({selected_year})")
        if os.path.exists(min_gwl_path):
            st.image(min_gwl_path, use_container_width=True)
        else:
            st.warning("Image not found.")

    with col3:
        st.write(f"Recharge ({selected_year})")
        if os.path.exists(recharge_path):
            st.image(recharge_path, use_container_width=True)
        else:
            st.warning("Image not found.")
