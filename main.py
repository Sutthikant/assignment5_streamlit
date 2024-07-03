import streamlit as st
import pandas as pd
import numpy as np
from model.model import query

st.title('Cat Breed Search')

title = st.text_input("Cat Breed", "Siamese")

if st.button("Find Image"):
    st.write(f"Finding image for {title}...")
    image = query(title)
    st.image(image, caption=title)