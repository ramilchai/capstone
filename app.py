import streamlit as st
from PIL import Image
import pandas as pd

#######################################################
# Page Title
#######################################################

img_header = Image.open('images\meet-your-next-favourite-book31.jpg')

st.image(img_header, use_column_width=True )

st.write("""

# Comic Books Recommendation
## Based on preferences in fantasy books!
 
***
""")