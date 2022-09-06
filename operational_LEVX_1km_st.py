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
prec_ml = (pd.DataFrame(alg["pipe"].predict_proba(model_x_var))).iloc[:,1].map("{:.0%}".format).values

#open algorithm skycover d0
alg = pickle.load(open("algorithms/skyc1_LEVX_1km_d0.al","rb"))
 
#select x _var
model_x_var = meteo_model[:24][alg["x_var"]]

#forecast machine learning skycover
skyc1_ml = alg["pipe"].predict(model_x_var)

#open algorithm cloud height d0
alg = pickle.load(open("algorithms/skyl1_LEVX_1km_d0.al","rb"))
 
#select x _var
model_x_var = meteo_model[:24][alg["x_var"]]

#forecast machine learning cloud height
skyl1_ml = alg["pipe"].predict(model_x_var)

#open algorithm brfg d0
alg = pickle.load(open("algorithms/brfg_LEVX_1km_d0.al","rb"))
 
#select x _var
model_x_var = meteo_model[:24][alg["x_var"]]

#forecast machine learning brfg
brfg_ml = alg["pipe"].predict(model_x_var)

#show results prec visibility fog cloud cover
st.write("#### **Machine learning results (Horizontal visibility, BR/FG, cloud low layer cover and height) forecast D0**")
st.write("###### **Horizontal visibility min (T-1hour,T]**")
st.write("###### **Precipitation, BR or FG, cloud cover and cloud height on time T**")

df_for0=pd.DataFrame({"time UTC":meteo_model[:24].index,
                     "visibility <=1000m (prob)":vis_ml,
                     "Precipitation (prob)":prec_ml,
                     "Cloud cover":skyc1_ml,
                     "Cloud height":skyl1_ml,
                     "Fog or BR":brfg_ml})

df_all=pd.concat([df_for0.set_index("time UTC"),metar_df],axis=1).reset_index()
df_all=df_all.rename(columns={"index": "Time UTC"})
AgGrid(df_all)


#show results wind and temperature

#open algorithm dir d0
alg = pickle.load(open("algorithms/dir_LEVX_1km_d0.al","rb"))
 
#select x _var
model_x_var = meteo_model[:24][alg["x_var"]]

#forecast machine learning brfg
dir_ml = alg["pipe"].predict(model_x_var)

st.write("#### **Results wind and temperature forecast  D0**")
st.write("###### **Wind speed mean interval [T-1hour,T)**")
st.write("###### **Wind gust, direction and temperature on time T**")         
df_for0 = pd.DataFrame({"time UTC":meteo_model[:24].index,
                     "Wind direction":dir_ml,
                     "dir WRF":round(model_x_var["dir0"],0)})

df_all = pd.concat([df_for0.set_index("time UTC"),metar_df],axis=1).reset_index()
df_all = df_all.rename(columns={"index": "Time UTC"})
AgGrid(df_all)

st.write("Project [link](https://github.com/granantuin/LEVX_1km)")
