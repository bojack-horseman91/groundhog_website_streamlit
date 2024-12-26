import streamlit as st
from pathlib import Path

def navbar():
    # Path to the image
    image_path = Path("assets/images/groundhog.jpg")
    
    # Check if the image exists
    if not image_path.exists():
        st.error(f"Image not found: {image_path}")
        return

    # Convert the image to a base64 string
    import base64
    with open(image_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    # Navbar with the image
    st.markdown(f"""
    <style>
        .navbar {{
            background-color: #2c3e50;
            color: white;
            padding: 15px;
            font-size: 22px;
            display: flex;
            align-items: center;
        }}
        .navbar img {{
            height: 50px;
            margin-right: 15px;
            border-radius: 50%; /* Makes the avatar circular */
        }}
        .navbar h1 {{
            margin: 0;
            color: white;
            display: inline;
        }}
        .navbar h2 {{
            margin: 0;
            color: #ecf0f1;
            font-size: 16px;
        }}
    </style>
    <div class="navbar">
        <img src="data:image/jpeg;base64,{img_base64}" alt="Groundhog Avatar">
        <div>
            <h1>GroundHog</h1>
            <h2>Revolutionizing Groundwater Downscaling</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

