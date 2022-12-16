import numpy as np
import pandas as pd
from help_functions import get_meteogalicia_model_1Km, get_metar
import pickle
import streamlit as st
from st_aggrid import AgGrid

st.set_page_config(page_title="Vigo airport (LEVX) forecast Machine Learning",layout="wide")


#open algorithm visibility d0 d1
alg = pickle.load(open("algorithms/brfg_LEVX_1km_time_d0.al","rb"))
alg1 = pickle.load(open("algorithms/brfg_LEVX_1km_time_d1.al","rb"))

#load raw meteorological model and add time variables
meteo_model,con = get_meteogalicia_model_1Km (alg["coor"])
meteo_model["hour"] = meteo_model.index.hour
meteo_model["month"] = meteo_model.index.month
meteo_model["dayofyear"] = meteo_model.index.dayofyear
meteo_model["weekofyear"] = meteo_model.index.weekofyear

#select x _var
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]
brfg_ml = alg["pipe"].predict(model_x_var)
brfg_ml1 = alg1["pipe"].predict(model_x_var1)


#open algorithm temp d0 d1
alg = pickle.load(open("algorithms/temp_LEVX_1km_time_d0.al","rb"))
alg1 = pickle.load(open("algorithms/temp_LEVX_1km_time_d1.al","rb"))
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]
temp_ml = alg["pipe"].predict(model_x_var)
temp_ml1 = alg1["pipe"].predict(model_x_var1)

#open algorithm dir d0 d1
alg = pickle.load(open("algorithms/dir_LEVX_1km_time_d0.al","rb"))
alg1 = pickle.load(open("algorithms/dir_LEVX_1km_time_d1.al","rb"))
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]
dir_ml = alg["pipe"].predict(model_x_var)
dir_ml1 = alg1["pipe"].predict(model_x_var1)

#open algorithm spd d0 d1
alg = pickle.load(open("algorithms/spd_LEVX_1km_time_d0.al","rb"))
alg1 = pickle.load(open("algorithms/spd_LEVX_1km_time_d1.al","rb"))
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]
spd_ml = alg["pipe"].predict(model_x_var)
spd_ml1 = alg["pipe"].predict(model_x_var1)

#open algorithm gust d0 d1
alg = pickle.load(open("algorithms/gust_LEVX_1km_time_d0.al","rb"))
alg1 = pickle.load(open("algorithms/gust_LEVX_1km_time_d1.al","rb"))
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]
gust_ml = alg["pipe"].predict(model_x_var)
gust_ml1 = alg1["pipe"].predict(model_x_var1)

#open algorithm tempd d0 d1
alg = pickle.load(open("algorithms/tempd_LEVX_1km_time_d0.al","rb"))
alg1 = pickle.load(open("algorithms/tempd_LEVX_1km_time_d1.al","rb"))
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]
tempd_ml = alg["pipe"].predict(model_x_var)
tempd_ml1 = alg1["pipe"].predict(model_x_var1)

#open algorithm H visibility d0 d1
alg = pickle.load(open("algorithms/vis_LEVX_1km_time_d0.al","rb"))
alg1 = pickle.load(open("algorithms/vis_LEVX_1km_time_d1.al","rb"))
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]
vis_ml = alg["pipe"].predict(model_x_var)
vis_ml1 = alg1["pipe"].predict(model_x_var1)


#open algorithm precipitation d0 d1
alg = pickle.load(open("algorithms/prec_LEVX_1km_time_d0.al","rb"))
alg1 = pickle.load(open("algorithms/prec_LEVX_1km_time_d1.al","rb"))
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]
prec_ml = alg["pipe"].predict(model_x_var)
prec_ml1 = alg["pipe"].predict(model_x_var1)

#open algorithm pres d0 d1
alg = pickle.load(open("algorithms/pres_LEVX_1km_time_d0.al","rb"))
alg1 = pickle.load(open("algorithms/pres_LEVX_1km_time_d1.al","rb"))
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]
pres_ml = alg["pipe"].predict(model_x_var)
pres_ml1 = alg["pipe"].predict(model_x_var1)

#open algorithm skyc1 d0 d1
alg = pickle.load(open("algorithms/skyc1_LEVX_1km_time_d0.al","rb"))
alg1 = pickle.load(open("algorithms/skyc1_LEVX_1km_time_d1.al","rb"))
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]
skyc1_ml = alg["pipe"].predict(model_x_var)
skyc1_ml1 = alg["pipe"].predict(model_x_var1)

#open algorithm skyl1 d0 d1
alg = pickle.load(open("algorithms/skyl1_LEVX_1km_time_d0.al","rb"))
alg1 = pickle.load(open("algorithms/skyl1_LEVX_1km_time_d1.al","rb"))
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]
skyl1_ml = alg["pipe"].predict(model_x_var)
skyl1_ml1 = alg["pipe"].predict(model_x_var1)

#get metar today
try:
  metar_df=get_metar("LEVX",con)
except:
  metar_df = pd.DataFrame()

st.write("###### **BR or FG, Cloud (cover and height), temperature and dew temperature, (WRF:meteorological model, ml: machine learning)**")
df_for0=pd.DataFrame({"time UTC": meteo_model[:48].index,
                      "Fog/BR ml": np.concatenate((brfg_ml,brfg_ml1),axis=0),
                      "Cloud cover ml": np.concatenate((skyc1_ml,skyc1_ml1),axis=0),
                      "Cloud height ml": np.concatenate((skyl1_ml,skyl1_ml1),axis=0),
                      "Temperature WRF":np.concatenate((np.rint(model_x_var["temp0"]-273.16),np.rint(model_x_var1["temp0"]-273.16)),axis=0),
                      "Temperature ml":np.concatenate((np.rint(temp_ml-273.16),np.rint(temp_ml1-273.16)),axis=0),
                      "Dew T ml":np.concatenate((np.rint(tempd_ml-273.16),np.rint(tempd_ml1-273.16)),axis=0)})

df_all=pd.concat([df_for0.set_index("time UTC"),metar_df],axis=1).reset_index()
df_all=df_all.rename(columns={"index": "Time UTC"})
AgGrid(df_all)

st.write("###### **Wind gust, intensity and direction, (WRF:meteorological model, ml: machine learning)**")  
df_for0 = pd.DataFrame({"time UTC":meteo_model[:48].index,
                        "dir ml":np.concatenate((dir_ml,dir_ml1),axis=0),
                        "dir WRF":np.concatenate((np.rint(model_x_var["dir0"]),np.rint(model_x_var1["dir0"])),axis=0),
                        "spd WRF": np.concatenate((np.rint(model_x_var["mod0"]*1.94384),np.rint(model_x_var1["mod0"]*1.94384)),axis=0),
                        "spd ml": np.concatenate((np.rint(spd_ml*1.94384),np.rint(spd_ml1*1.94384)),axis =0),
                        "gust ml": np.concatenate((gust_ml,gust_ml1),axis=0),
                        "gust WRF":np.concatenate((round(model_x_var["wind_gust0"]*1.94384,0),round(model_x_var1["wind_gust0"]*1.94384,0)),axis=0)})

df_all = pd.concat([df_for0.set_index("time UTC"),metar_df],axis=1).reset_index()
df_all = df_all.rename(columns={"index": "Time UTC"})
AgGrid(df_all)

st.write("###### **Horizontal visibility, precipitation, QNH, (WRF:meteorological model, ml: machine learning)**")
df_for0 = pd.DataFrame({"time UTC":meteo_model[:48].index,
                        "vis Hor ml":np.concatenate((vis_ml,vis_ml1),axis=0),
                        "prec WRF": np.concatenate((round(model_x_var["prec0"],1),round(model_x_var1["prec0"],1)),axis=0),
                        "prec ml": np.concatenate((prec_ml,prec_ml1),axis=0),
                        "QNH WRF":np.concatenate((np.rint(model_x_var["mslp0"]/100),np.rint(model_x_var1["mslp0"]/100)),axis=0),
                        "QNH ml": np.concatenate((np.rint(pres_ml),np.rint(pres_ml1)),axis=0)})

df_all = pd.concat([df_for0.set_index("time UTC"),metar_df],axis=1).reset_index()
df_all = df_all.rename(columns={"index": "Time UTC"})
AgGrid(df_all)


st.write("Project [link](https://github.com/granantuin/LEVX_1km)")
