import numpy as np
import pandas as pd
from help_functions import get_meteogalicia_model_1Km, get_metar
import pickle
import streamlit as st
from st_aggrid import AgGrid

st.set_page_config(page_title="LEVX Machine Learning",layout="wide")

#get metar today
try:
  metar_df=get_metar("LEVX")
except:
  metar_df = pd.DataFrame()

#open algorithm visibility d0
alg = pickle.load(open("algorithms/vis_LEVX_1km_d0.al","rb"))

#load raw meteorological model and get model variables
meteo_model = get_meteogalicia_model_1Km (alg["coor"])
 
#select x _var
model_x_var = meteo_model[:24][alg["x_var"]]

#forecast machine learning  horizontal visibility meters
vis_ml = (pd.DataFrame(alg["pipe"].predict_proba(model_x_var))).iloc[:,0].map("{:.0%}".format).values

#open algorithm precipitation d0
alg = pickle.load(open("algorithms/prec_LEVX_1km_d0.al","rb"))
 
#select x _var
model_x_var = meteo_model[:24][alg["x_var"]]

#forecast machine learning  horizontal visibility meters
prec_ml = (pd.DataFrame(alg["pipe"].predict_proba(model_x_var))).iloc[:,0].map("{:.0%}".format).values

#open algorithm precipitation d0
alg = pickle.load(open("algorithms/skyc1_LEVX_1km_d0.al","rb"))
 
#select x _var
model_x_var = meteo_model[:24][alg["x_var"]]

#forecast machine learning  horizontal visibility meters
skyc1_ml = alg["pipe"].predict(model_x_var)

#show results prec visibility fog cloud cover
st.write("#### **Machine learning results (Horizontal visibility, BR/FG, cloud low layer cover and height) forecast D0**")
st.write("###### **Horizontal visibility min (T-1hour,T]**")
st.write("###### **Precipitation, BR or FG, cloud cover and cloud height on time T**")

"""
df_for0=pd.DataFrame({"time UTC":meteo_model[:24].index,
                     "visibility <=1000m (prob)":vis_ml,
                     "Precipitation (prob)":prec_ml,
                     "Fog or BR":brfg_ml,
                     "Cloud cover":skyc1_ml,
                     "Cloud height":skyl1_ml})
"""                     

df_for0=pd.DataFrame({"time UTC":meteo_model[:24].index,
                     "visibility <=1000m (prob)":vis_ml,
                     "Precipitation (prob)":prec_ml,
                     "Cloud cover":skyc1_ml,})

df_all=pd.concat([df_for0.set_index("time UTC"),metar_df],axis=1).reset_index()
df_all=df_all.rename(columns={"index": "Time UTC"})
AgGrid(df_all)


