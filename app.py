import streamlit as st
from components.navbar import navbar
from components.upload_page import upload_and_process_page
from components.visualization_page import visualization_page
from components.image_view import display_images
from components.about import show_contact_information
def main():
    # Display Navbar
    navbar()
    
    def page1():
        with st.spinner("loading page........"):
            upload_and_process_page()
    def page2():
        with st.spinner("loading page........"):
            visualization_page()
    def page3():
        with st.spinner("loading page........"):
            display_images()
    def page4():
        with st.spinner("loading page........"):
            show_contact_information()
    # Sidebar Navigation
    page=st.navigation([st.Page(page1,title="Upload and Downscale"),
                        st.Page(page2,title="Interactive plot of year 22")
                        ,st.Page(page3,title="Images of Upsampled GWL all years"),
                        st.Page(page4,title="Contact us")
                        ],expanded=True)
    st.sidebar.title("Navigation Tab")
    page.run()
    st.snow()
    

if __name__ == "__main__":
    main()
