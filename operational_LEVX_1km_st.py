import numpy as np
import pandas as pd
from help_functions import get_meteogalicia_model, get_metar, get_table_download_link
import pickle
import streamlit as st
import plotly.express as px
from st_aggrid import AgGrid

st.set_page_config(page_title="LEVX Machine Learning",layout="wide")
#open algorithm
alg=pickle.load(open("algorithms/vis_LEVX_d0.al","rb"))

#load raw meteorological model and get model variables
meteo_model=get_meteogalicia_model(alg["coor"])


#map Vigo airport
if st.checkbox("model points map?"):
  #map
  st.write("#### **Vigo airport and WRF Meteogalicia model**") 
  px.set_mapbox_access_token("pk.eyJ1IjoiZ3JhbmFudHVpbiIsImEiOiJja3B4dGU4OTkwMTFmMm9ycnNhMjJvaGJqIn0.VWzx_PkD9A5cSUVsn_ijCA")
  dist_map=px.scatter_mapbox(alg["coor"], hover_data=['distance'],lat='lat', lon='lon',color='distance',
                             color_continuous_scale=px.colors.cyclical.IceFire,)
  st.plotly_chart(dist_map)

 
#get metar today
try:
  metar_df=get_metar("LEVX")
except:
  metar_df = pd.DataFrame()

#select x _var
model_x_var=meteo_model[:24][alg["x_var"]]

#forecast machine learning  horizontal visibility meters
vis_ml=(pd.DataFrame(alg["ml_model"].predict_proba(model_x_var))).iloc[:,0].map("{:.0%}".format).values
