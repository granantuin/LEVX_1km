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
alg = pickle.load(open("algorithms/brfg_LEVX_1km_time_d0.al","rb"))


#load raw meteorological model and add time variables
meteo_model = get_meteogalicia_model_1Km (alg["coor"])
meteo_model["hour"] = meteo_model.index.hour
meteo_model["month"] = meteo_model.index.month
meteo_model["dayofyear"] = meteo_model.index.dayofyear
meteo_model["weekofyear"] = meteo_model.index.weekofyear

 
#select x _var
model_x_var = meteo_model[:24][alg["x_var"]]

#forecast machine learning brfg
brfg_ml = alg["pipe"].predict(model_x_var)

#open algorithm temp d0
alg = pickle.load(open("algorithms/temp_LEVX_1km_time_d0.al","rb"))

#select x _var
model_x_var = meteo_model[:24][alg["x_var"]]

#forecast machine learning temp
temp_ml = alg["pipe"].predict(model_x_var)

#open algorithm dir d0
alg = pickle.load(open("algorithms/dir_LEVX_1km_time_d0.al","rb"))

#select x _var
model_x_var = meteo_model[:24][alg["x_var"]]

#forecast machine learning dir
dir_ml = alg["pipe"].predict(model_x_var)

#open algorithm dir d0
alg = pickle.load(open("algorithms/spd_LEVX_1km_time_d0.al","rb"))

#select x _var
model_x_var = meteo_model[:24][alg["x_var"]]

#forecast machine learning dir
spd_ml = alg["pipe"].predict(model_x_var)

#open algorithm dir d0
alg = pickle.load(open("algorithms/gust_LEVX_1km_time_d0.al","rb"))

#select x _var
model_x_var = meteo_model[:24][alg["x_var"]]

#forecast machine learning dir
gust_ml = alg["pipe"].predict(model_x_var)

#open algorithm tempd d0
alg = pickle.load(open("algorithms/tempd_LEVX_1km_time_d0.al","rb"))

#select x _var
model_x_var = meteo_model[:24][alg["x_var"]]

#forecast machine learning tempd
tempd_ml = alg["pipe"].predict(model_x_var)

#open algorithm H visibility d0
alg = pickle.load(open("algorithms/vis_LEVX_1km_time_d0.al","rb"))

#select x _var
model_x_var = meteo_model[:24][alg["x_var"]]

#forecast machine learning visibility
vis_ml = alg["pipe"].predict(model_x_var)

#open algorithm precipitation d0
alg = pickle.load(open("algorithms/prec_LEVX_1km_time_d0.al","rb"))

#select x _var
model_x_var = meteo_model[:24][alg["x_var"]]

#forecast machine learning prec
prec_ml = alg["pipe"].predict(model_x_var)

#open algorithm pres d0
alg = pickle.load(open("algorithms/pres_LEVX_1km_time_d0.al","rb"))

#select x _var
model_x_var = meteo_model[:24][alg["x_var"]]

#forecast machine learning prec
pres_ml = alg["pipe"].predict(model_x_var)

#open algorithm skyc1 d0
alg = pickle.load(open("algorithms/skyc1_LEVX_1km_time_d0.al","rb"))

#select x _var
model_x_var = meteo_model[:24][alg["x_var"]]

#forecast machine learning prec
skyc1_ml = alg["pipe"].predict(model_x_var)




st.write("###### **BR or FG, Skycover, temperature and dew temperature, (WRF:meteorological model, ml: machine learning **")

df_for0=pd.DataFrame({"time UTC": meteo_model[:24].index,
                      "Fog/BR ml": brfg_ml,
                      "Skycover ml": skyc1_ml,
                      "Temperature WRF":round(model_x_var["temp0"]-273.16,0),
                      "Temperature ml":np.rint(temp_ml-273.16),
                      "Dew T ml":np.rint(tempd_ml-273.16)})

df_all=pd.concat([df_for0.set_index("time UTC"),metar_df],axis=1).reset_index()
df_all=df_all.rename(columns={"index": "Time UTC"})
AgGrid(df_all)

st.write("###### **Wind gust, intensity and  direction on time T**")  

df_for0 = pd.DataFrame({"time UTC":meteo_model[:24].index,
                        "dir ml":dir_ml,
                        "dir WRF":round(model_x_var["dir0"],0),
                        "spd WRF":round(model_x_var["mod0"]*1.94384,0),
                        "spd ml": np.rint(spd_ml*1.94384),
                        "gust ml": gust_ml,
                        "gust WRF":round(model_x_var["wind_gust0"]*1.94384,0)})

df_all = pd.concat([df_for0.set_index("time UTC"),metar_df],axis=1).reset_index()
df_all = df_all.rename(columns={"index": "Time UTC"})
AgGrid(df_all)

st.write("###### **Horizontal visibility, Precipitation on time T**")

df_for0 = pd.DataFrame({"time UTC":meteo_model[:24].index,
                        "vis Hor ml":vis_ml,
                        "prec WRF": round(model_x_var["prec0"],1),
                        "prec ml": prec_ml,
                        "QNH WRF":np.rint(model_x_var["mslp0"]/100),
                        "QNH ml": np.rint(pres_ml)})

df_all = pd.concat([df_for0.set_index("time UTC"),metar_df],axis=1).reset_index()
df_all = df_all.rename(columns={"index": "Time UTC"})
AgGrid(df_all)


st.write("Project [link](https://github.com/granantuin/LEVX_1km)")
