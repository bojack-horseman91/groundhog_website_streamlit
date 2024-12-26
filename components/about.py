import streamlit as st

def show_contact_information():
    # Define contact details
    first_author_details = {
        "name": "Saleh Sakib Ahmed",
        "email": "birdhunterx91 at gmail dot com",
        "role": "First Author & Developer",
        "googe_scholars": "https://scholar.google.com/citations?user=t49Bx_gAAAAJ&hl=en",
    }

    corresponding_author_details = {
        "name": "Dr. M. Sohel Rahman",
        "email": "msrahman at cse dot buet dot ac dot org",
        "role": "Corresponding Author",
        "googe_scholars": "https://scholar.google.com/citations?user=IUwFD9gAAAAJ&hl=en",
    }

    paper_link = "https://doi.org/example-paper-link"

    # Display contact details using Streamlit
    st.header("Contact Information")

    # First Author & Developer Section
    st.subheader(f"{first_author_details['role']}")
    st.write(f"**Name:** {first_author_details['name']}")
    a=f"**Email:** [{first_author_details['email']}]"
    st.write(a)
    st.markdown(f"**Google Scholars**:[{first_author_details['name']}]({first_author_details['googe_scholars']})", unsafe_allow_html=True)


    # Separator
    st.markdown("---")

    # Corresponding Author Section
    st.subheader(f"{corresponding_author_details['role']}")
    st.write(f"**Name:** {corresponding_author_details['name']}")
    st.write(f"**Email:** [{corresponding_author_details['email']}]")
    st.markdown(f"**Google Scholars**:[{corresponding_author_details['name']}]({corresponding_author_details['googe_scholars']})", unsafe_allow_html=True)

    # Separator
    st.markdown("---")

    # Paper Link Section
    st.subheader("Read the Paper")
    st.markdown(f"[Click here to view the paper]({paper_link})", unsafe_allow_html=True)

