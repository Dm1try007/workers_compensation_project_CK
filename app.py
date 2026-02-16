import streamlit as st

st.set_page_config(page_title="Workers Compensation", layout="wide")

pages = [
    st.Page("analysis_and_model.py", title="Анализ и модель"),
    st.Page("presentation.py", title="Презентация"),
]

current_page = st.navigation(pages, position="sidebar", expanded=True)
current_page.run()
