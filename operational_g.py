#@title Get metars and meteorological model
import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta
from io import BytesIO
import base64
import pickle
import warnings
import streamlit as st
import matplotlib.pyplot as plt
from st_aggrid import AgGrid

warnings.filterwarnings("ignore")

def get_metar(oaci,control):
     """
     get metar from IOWA university database
     
     in: OACI airport code
     Returns
      -------
     dataframe with raw metar.
     """
     #today metar control =True
     if control:
       today = pd.to_datetime("today")+timedelta(1)
       yes = today-timedelta(1)
     else:
        today = pd.to_datetime("today")+timedelta(1)
        yes = today-timedelta(2)

     #url string
     s1="https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station="
     s2="&data=all"
     s3="&year1="+yes.strftime("%Y")+"&month1="+yes.strftime("%m")+"&day1="+yes.strftime("%d")
     s4="&year2="+today.strftime("%Y")+"&month2="+today.strftime("%m")+"&day2="+today.strftime("%d")
     s5="&tz=Etc%2FUTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=1&report_type=2"
     url=s1+oaci+s2+s3+s4+s5
     df_metar_global=pd.read_csv(url,parse_dates=["valid"],).rename({"valid":"time"},axis=1)
     df_metar = df_metar_global[["time",'tmpf', 'dwpf','drct', 'sknt', 'alti','vsby',
                                 'gust', 'skyc1', 'skyc2', 'skyl1', 'skyl2','wxcodes',
                                 "metar"]].set_index("time")
     
     #temperature dry a dew point to celsius                            
     df_metar["temp_o"] = np.rint((df_metar.tmpf - 32)*5/9)
     df_metar["tempd_o"] = np.rint((df_metar.dwpf - 32)*5/9)

     #QNH to mb
     df_metar["mslp_o"] = np.rint(df_metar.alti*33.8638)

     #visibility SM to meters
     df_metar["visibility_o"] =np.rint(df_metar.vsby/0.00062137)

     #wind direction, intensity and gust
     df_metar["spd_o"] = df_metar["sknt"]
     df_metar["dir_o"] = df_metar["drct"]
     df_metar['gust_o'] = df_metar['gust'] 

     #Add suffix cloud cover and cloud height, present weather, and metar 
     df_metar['skyc1_o'] = df_metar['skyc1']
     df_metar["skyl1_o"] = df_metar["skyl1"]
     df_metar['skyc2_o'] = df_metar['skyc2']
     df_metar["skyl2_o"] = df_metar["skyl2"]
     df_metar["wxcodes_o"] = df_metar["wxcodes"]
     df_metar["metar_o"] = df_metar["metar"]
     

     # Select all columns that do not start with "_o"
     columns_to_keep = [col for col in df_metar.columns if col.endswith("_o")]
     df_metar = df_metar[columns_to_keep] 

     return df_metar 



  
def get_meteogalicia_model_1Km(coorde):
    """
    get meteogalicia model (1Km)from algo coordenates
    Returns
    -------
    dataframe with meteeorological variables forecasted.
    """
    
    #defining url to get model from Meteogalicia server
    #today = C
    #today = pd.to_datetime("today")+timedelta(1)
    today = pd.to_datetime("today")

    try:

      head1 = "http://mandeo.meteogalicia.es/thredds/ncss/wrf_1km_baixas/fmrc/files/"
      head2 = today.strftime("/%Y%m%d/wrf_arw_det1km_history_d05")
      head3 = today.strftime("_%Y%m%d_0000.nc4?")
      head = head1+head2+head3
  
      var1="var=dir&var=mod&var=wind_gust&var=mslp&var=temp&var=rh&var=visibility&var=lhflx"
      var2="&var=lwflx&var=conv_prec&var=prec&var=swflx&var=shflx&var=cape&var=cin&var=cfh&var=T850"
      var3="&var=cfl&var=cfm&var=cft&var=HGT500&var=HGT850&var=T500&var=snow_prec&var=snowlevel"
      var=var1+var2+var3
  
      f_day=(today+timedelta(days=2)).strftime("%Y-%m-%d") 
      tail="&time_start="+today.strftime("%Y-%m-%d")+"T01%3A00%3A00Z&time_end="+f_day+"T23%3A00%3A00Z&accept=csv"
  
      dffinal=pd.DataFrame() 
      for coor in list(zip(coorde.lat.tolist(),coorde.lon.tolist(),np.arange(0,len(coorde.lat.tolist())).astype(str))):
          dffinal=pd.concat([dffinal,pd.read_csv(head+var+"&latitude="+str(coor[0])+"&longitude="+str(coor[1])+tail,).add_suffix(str(coor[2]))],axis=1)    
  
      
      #filter all columns with lat lon and date
      dffinal=dffinal.filter(regex='^(?!(lat|lon|date).*?)')
  
      #remove column string between brakets
      new_col=[c.split("[")[0]+c.split("]")[-1] for c in dffinal.columns]
      for col in zip(dffinal.columns,new_col):
          dffinal=dffinal.rename(columns = {col[0]:col[1]})
  
      dffinal=dffinal.set_index(pd.date_range(start=today.strftime("%Y-%m-%d"), end=(today+timedelta(days=3)).strftime("%Y-%m-%d"), freq="H")[1:-1])  
      control = True
    except:

      today = pd.to_datetime("today")-timedelta(1)
      head1 = "http://mandeo.meteogalicia.es/thredds/ncss/wrf_1km_baixas/fmrc/files/"
      head2 = today.strftime("/%Y%m%d/wrf_arw_det1km_history_d05")
      head3 = today.strftime("_%Y%m%d_0000.nc4?")
      head = head1+head2+head3
  
      var1="var=dir&var=mod&var=wind_gust&var=mslp&var=temp&var=rh&var=visibility&var=lhflx"
      var2="&var=lwflx&var=conv_prec&var=prec&var=swflx&var=shflx&var=cape&var=cin&var=cfh&var=T850"
      var3="&var=cfl&var=cfm&var=cft&var=HGT500&var=HGT850&var=T500&var=snow_prec&var=snowlevel"
      var=var1+var2+var3
  
      f_day=(today+timedelta(days=2)).strftime("%Y-%m-%d") 
      tail="&time_start="+today.strftime("%Y-%m-%d")+"T01%3A00%3A00Z&time_end="+f_day+"T23%3A00%3A00Z&accept=csv"
  
      dffinal=pd.DataFrame() 
      for coor in list(zip(coorde.lat.tolist(),coorde.lon.tolist(),np.arange(0,len(coorde.lat.tolist())).astype(str))):
          dffinal=pd.concat([dffinal,pd.read_csv(head+var+"&latitude="+str(coor[0])+"&longitude="+str(coor[1])+tail,).add_suffix(str(coor[2]))],axis=1)    
  
      
      #filter all columns with lat lon and date
      dffinal=dffinal.filter(regex='^(?!(lat|lon|date).*?)')
  
      #remove column string between brakets
      new_col=[c.split("[")[0]+c.split("]")[-1] for c in dffinal.columns]
      for col in zip(dffinal.columns,new_col):
          dffinal=dffinal.rename(columns = {col[0]:col[1]})
  
      dffinal=dffinal.set_index(pd.date_range(start=today.strftime("%Y-%m-%d"), end=(today+timedelta(days=3)).strftime("%Y-%m-%d"), freq="H")[1:-1])  
      control= False  

     
    return dffinal , control


# Set the directory you want to list algorithms filenames from
algo_dir = 'algorithms/'

#get meteorological model from algorithm file. Select "coor" key to get coordinates. Pick up first algorithm all same coordinates
meteo_model,con = get_meteogalicia_model_1Km(pickle.load(open(algo_dir+os.listdir(algo_dir)[0],"rb"))["coor"])

#add time variables
meteo_model["hour"] = meteo_model.index.hour
meteo_model["month"] = meteo_model.index.month
meteo_model["dayofyear"] = meteo_model.index.dayofyear
meteo_model["weekofyear"] = meteo_model.index.isocalendar().week.astype(int)

#show meteorological model and control variable. Control variable True if Day analysis = today 
#st.write(#### **Day analysis = today :**",con)
#st.write(meteo_model)

metars = get_metar("LEVX",con)
#AgGrid(metars)


#@title Wind intensity
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

#open algorithm spd d0 d1
alg = pickle.load(open("algorithms/spd_LEVX_1km_time_d0.al","rb"))
alg1 = pickle.load(open("algorithms/spd_LEVX_1km_time_d1.al","rb"))

#select model variables
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]

# forecat spd from ml
spd_ml = alg["pipe"].predict(meteo_model[:24][alg["x_var"]])
spd_ml1 = alg1["pipe"].predict(meteo_model[24:48][alg1["x_var"]])

df_for = pd.DataFrame({"time":meteo_model[:48].index,
                       "spd_WRF": np.concatenate((np.rint(model_x_var["mod0"]*1.94384),
                                                   np.rint(model_x_var1["mod0"]*1.94384)),axis=0),
                       "spd_ml": np.concatenate((np.rint(spd_ml*1.94384),
                                                  np.rint(spd_ml1*1.94384)),axis =0),})
df_for = df_for.set_index("time")

#metar versus forecast
# set the max columns to none
pd.set_option('display.max_rows', 100)

# concat metars an forecast
df_res = pd.concat([df_for,metars["spd_o"]],axis = 1)

#get mae
df_res_dropna = df_res.dropna()
mae_ml = round(mean_absolute_error(df_res_dropna.spd_o,df_res_dropna.spd_ml),2)
mae_wrf = round(mean_absolute_error(df_res_dropna.spd_o,df_res_dropna.spd_WRF),2)

#print results
st.markdown("**Wind intensity knots**")
st.markdown("Reference (48 hours) Mean absolute error meteorological model: 1.35")
st.markdown("Reference (48 hours) Mean absolute error machine learning: 0.89")

#show results
fig, ax = plt.subplots(figsize=(10,6))
df_res.dropna().plot(grid=True, ax=ax, linestyle='--');
title = "Mean absolute error meteorological model: {}\nMean absolute error machine learning: {} ".format(mae_wrf,mae_ml)
ax.set_title(title)

# Display the plot in Streamlit
st.pyplot(fig)

# Create the plot
fig, ax = plt.subplots(figsize=(10,6))
df_for.plot(grid=True, ax=ax, linestyle='--')
ax.set_title("Forecast meteorological model versus machine learning")

# Display the plot in Streamlit
st.pyplot(fig)

#@title Wind direction
from sklearn.metrics import accuracy_score

#open algorithm spd d0 d1
alg = pickle.load(open("algorithms/dir_LEVX_1km_time_d0.al","rb"))
alg1 = pickle.load(open("algorithms/dir_LEVX_1km_time_d1.al","rb"))

#select model variables
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]

# forecat spd from ml
dir_ml = alg["pipe"].predict(model_x_var)
dir_ml1 = alg1["pipe"].predict(model_x_var1)

#set up dataframe forecast machine learning and WRF
df_for = pd.DataFrame({"time":meteo_model[:48].index,
                        "dir_WRF": np.concatenate((model_x_var["dir0"],model_x_var1["dir0"]),axis=0),
                        "dir_ml": np.concatenate((dir_ml,dir_ml1),axis =0),})
df_for = df_for.set_index("time")

#label dir_o and dir0 .wind direction to interval dir=-1 variable wind
interval = pd.IntervalIndex.from_tuples([(-1.5, -0.5),(-0.5,20), (20, 40), (40, 60),
                                           (60,80),(80,100),(100,120),(120,140),(140,160),
                                           (160,180),(180,200),(200,220),(220,240),
                                           (240,260),(260,280),(280,300),(300,320),
                                           (320,340),(340,360)])

labels = ['VRB', '[0, 20]', '(20, 40]', '(40, 60]','(60, 80]', '(80, 100]',
          '(100, 120]', '(120, 140]','(140, 160]', '(160, 180]', '(180, 200]',
          '(200, 220]','(220, 240]', '(240, 260]', '(260, 280]', '(280, 300]',
          '(300, 320]', '(320, 340]', '(340, 360]']

df_for["dir_WRF_l"] = pd.cut(df_for["dir_WRF"], bins=interval,retbins=False,
                        labels=labels).map({a:b for a,b in zip(interval,labels)}).astype(str)

#dir_o to intervals 
metars["dir_o_l"] = pd.cut(metars["dir_o"].replace("M",-1).astype(float), bins=interval,retbins=False,
                        labels=labels).map({a:b for a,b in zip(interval,labels)}).astype(str)                    

#metar versus forecast
# set the max columns to none
pd.set_option('display.max_rows', 100)

# concat metars an forecast
df_res = pd.concat([df_for,metars[["dir_o","dir_o_l"]]],axis = 1)

#get accuracy
df_res_dropna = df_res.dropna()
acc_ml = round(accuracy_score(df_res_dropna.dir_o_l,df_res_dropna.dir_ml),2)
acc_wrf = round(accuracy_score(df_res_dropna.dir_o_l,df_res_dropna.dir_WRF_l),2)

#print results
st.markdown("**Wind direction**")
st.markdown("Reference (48 hours) Accuracy meteorological model: 0.20")
st.markdown("Reference (48 hours) Accuracy machine learning: 0.41")

#show results
fig, ax = plt.subplots(figsize=(10,6))
plt.plot(df_res_dropna.index, df_res_dropna['dir_ml'], marker="^", markersize=8, 
         markerfacecolor='w', linestyle='')
plt.plot(df_res_dropna.index, df_res_dropna['dir_o_l'], marker="*", markersize=13,
         markerfacecolor='k', linestyle='');
plt.plot(df_res_dropna.index, df_res_dropna['dir_WRF_l'], marker="v", markersize=8,
         markerfacecolor='w', linestyle='');
plt.legend(('direction ml', 'direction observed', 'direction WRF'),)
plt.grid(True)
plt.title("Accuracy meteorological model: {}\nAccuracy machine learning: {} ".format(acc_wrf,acc_ml))
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10,6))
plt.plot(df_for.index, df_for['dir_ml'],marker="^",linestyle='');
plt.plot(df_for.index, df_for['dir_WRF_l'],marker="v",linestyle='');
plt.legend(('direction ml','direction WRF'),)
plt.title("Forecast meteorological model versus machine learning")
plt.grid(True)
st.pyplot(fig)

#probabilistic results
prob = (np.concatenate((alg["pipe"].predict_proba(model_x_var),alg1["pipe"].predict_proba(model_x_var1)),axis =0)).transpose()
df_prob = pd.DataFrame(prob,index =alg["pipe"].classes_ ).T

# Find the columns where all values are less than or equal to 5%
cols_to_drop = df_prob.columns[df_prob.apply(lambda x: x <= 0.05).all()]
df_prob.drop(cols_to_drop, axis=1, inplace=True)
df_prob["time"] = meteo_model[:48].index

st.write("""Probabilistic results only columns more than 5%""")
AgGrid(round(df_prob,2))

#@title Wind gust
#open algorithm gust d0 d1
alg = pickle.load(open("algorithms/gust_LEVX_1km_time_d0.al","rb"))
alg1 = pickle.load(open("algorithms/gust_LEVX_1km_time_d1.al","rb"))

#select model variables
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]

# forecat gust from ml
gust_ml = alg["pipe"].predict(model_x_var)
gust_ml1 = alg1["pipe"].predict(model_x_var1)

#label metars gust data
metars["gust_o_l"] = ["No Gust" if c=="M" else "Gust" for c in metars.gust_o]

#set up dataframe forecast machine learning 
df_for = pd.DataFrame({"time":meteo_model[:48].index,
                       "gust_ml": np.concatenate((gust_ml,gust_ml1),axis =0),})
df_for = df_for.set_index("time")

# concat metars an forecast
df_res = pd.concat([df_for,metars["gust_o_l"]], axis = 1)
df_res_dropna = df_res.dropna()

#Heidke skill score
cm = pd.crosstab(df_res.dropna().gust_o_l, df_res.dropna().gust_ml, margins=True,)
HSS = 0
if cm.shape == (3,3):# complete confusion matrix to calculate HSS
  a = cm.values[0,0]
  b = cm.values[1,0]
  c = cm.values[0,1]
  d = cm.values[1,1]
  HSS = round(2*(a*d-b*c)/((a+c)*(c+d)+(a+b)*(b+d)),2)

#show results
st.markdown("**Wind gust**")
st.markdown("Reference (48 hours) Heidke skill Score: 0.42")
st.markdown("Confusion matrix")
st.write(cm)

fig, ax = plt.subplots(figsize=(10,6))
plt.plot(df_res_dropna.index, df_res_dropna['gust_ml'], marker="^", markersize=10, 
         markerfacecolor='w', linestyle='');
plt.plot(df_res_dropna.index, df_res_dropna['gust_o_l'],marker="*",linestyle='');
plt.legend(('gust ml', 'gust observed'),)
plt.grid(True)
plt.title("Heidke skill Score: {}".format(HSS))
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10,6))
plt.plot(df_for.index, df_for['gust_ml'],marker="^", markersize=8, markerfacecolor='w', linestyle='');
plt.title("Forecast machine learning")
plt.grid(True)
st.pyplot(fig)

#show probabilistic results
prob = (np.concatenate((alg["pipe"].predict_proba(model_x_var),alg1["pipe"].predict_proba(model_x_var1)),axis =0)).transpose()
df_prob = (pd.DataFrame(prob,index =alg["pipe"].classes_ ).T.set_index(meteo_model[:48].index))
df_prob["time"] = meteo_model[:48].index
st.write("""Probabilistic results""")
AgGrid(round(df_prob,2)) 

#@title Visibility
#open algorithm visibility d0 d1
#alg = pickle.load(open("algorithms/vis_LEVX_1km_time_d0_p.al","rb")) #_p for a plus algorithm
alg = pickle.load(open("algorithms/vis_LEVX_1km_time_d0.al","rb"))
alg1 = pickle.load(open("algorithms/vis_LEVX_1km_time_d1.al","rb"))

#select model variables
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]

# forecat vis from ml
vis_ml = alg["pipe"].predict(model_x_var)
vis_ml1 = alg1["pipe"].predict(model_x_var1)

#label metars vis data
metars["vis_o_l"] = ["<= 1000 m" if c<=1000 else "> 1000 m" for c in metars.visibility_o]

#label meteorological model visibility0
visibility0_l= ["<= 1000 m" if c<=1000 else "> 1000 m" for c in np.concatenate((model_x_var["visibility0"],model_x_var1["visibility0"]), axis=0)]

#set up dataframe forecast machine learning 
df_for = pd.DataFrame({"time":meteo_model[:48].index,
                       "vis_WRF": visibility0_l,
                       "vis_ml": np.concatenate((vis_ml,vis_ml1),axis =0),})
df_for = df_for.set_index("time")

# concat metars an forecast
df_res = pd.concat([df_for,metars["vis_o_l"]], axis = 1)
df_res_dropna = df_res.dropna()

#Heidke skill score ml
cm_ml = pd.crosstab(df_res.dropna().vis_o_l, df_res.dropna().vis_ml, margins=True,)
HSS_ml = 0
if cm_ml.shape == (3,3):# complete confusion matrix to calculate HSS
  a = cm_ml.values[0,0]
  b = cm_ml.values[1,0]
  c = cm_ml.values[0,1]
  d = cm_ml.values[1,1]
  HSS_ml = round(2*(a*d-b*c)/((a+c)*(c+d)+(a+b)*(b+d)),2)

#Heidke skill score meteorological model
cm_wrf = pd.crosstab(df_res.dropna().vis_o_l, df_res.dropna().vis_WRF, margins=True,)
HSS_wrf = 0
if cm_wrf.shape == (3,3):# complete confusion matrix to calculate HSS
  a = cm_wrf.values[0,0]
  b = cm_wrf.values[1,0]
  c = cm_wrf.values[0,1]
  d = cm_wrf.values[1,1]
  HSS_wrf = round(2*(a*d-b*c)/((a+c)*(c+d)+(a+b)*(b+d)),2)

#show results
st.markdown("**Visibility**")
st.markdown("Reference (48 hours) Heidke skill score meteorological model: 0.25")
st.markdown("Reference (48 hours) Heidke skill score machine learning: 0.52")
st.markdown("Confusion matrix machine learning")
st.write(cm_ml)
st.markdown("Confusion matrix meteorological model")
st.write(cm_wrf)

fig, ax = plt.subplots(figsize=(10,6))
plt.plot(df_res_dropna.index, df_res_dropna['vis_ml'],marker="^", markersize=8, 
         markerfacecolor='w', linestyle='');
plt.plot(df_res_dropna.index, df_res_dropna['vis_o_l'],marker="*",markersize=8, 
         markerfacecolor='w', linestyle='');
plt.plot(df_res_dropna.index, df_res_dropna['vis_WRF'],marker="v",markersize=8, 
         markerfacecolor='w', linestyle='');
plt.legend(('vis ml', 'vis observed',"vis WRF"),)
plt.grid(True)
plt.title("Heidke skill score meteorological model: {}\nHeidke skill score machine learning: {} ".format(HSS_wrf,HSS_ml))
st.pyplot(fig)


fig, ax = plt.subplots(figsize=(10,6))
plt.plot(df_for.index, df_for['vis_ml'],marker="^", markersize=8, 
         markerfacecolor='w', linestyle='');
plt.plot(df_for.index, df_for['vis_WRF'],marker="v",markersize=8, 
         markerfacecolor='w', linestyle='');
plt.title("Forecast machine learning")
plt.grid(True)
st.pyplot(fig)

#show probabilistic results
prob = (np.concatenate((alg["pipe"].predict_proba(model_x_var),alg1["pipe"].predict_proba(model_x_var1)),axis =0)).transpose()
df_prob = (pd.DataFrame(prob,index =alg["pipe"].classes_ ).T.set_index(meteo_model[:48].index))
df_prob["time"] = meteo_model[:48].index
st.write("""Probabilistic results""")
AgGrid(round(df_prob,2)) 

#@title BR or FG
#open algorithm prec d0 d1
alg = pickle.load(open("algorithms/brfg_LEVX_1km_time_d0.al","rb"))
alg1 = pickle.load(open("algorithms/brfg_LEVX_1km_time_d1.al","rb"))

#select model variables
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]

# forecat br/fg from ml
brfg_ml = alg["pipe"].predict(model_x_var)
brfg_ml1 = alg1["pipe"].predict(model_x_var1)

#label metars br/fg data
metars["brfg_o_l"] = "No BR/FG"
mask = metars['wxcodes_o'].str.contains("BR")
metars.loc[mask,["brfg_o_l"]] = "BR/FG"
mask = metars['wxcodes_o'].str.contains("FG")
metars.loc[mask,["brfg_o_l"]] = "BR/FG"



#set up dataframe forecast machine learning 
df_for = pd.DataFrame({"time": meteo_model[:48].index,
                       "brfg_ml": np.concatenate((brfg_ml,brfg_ml1),axis =0),})
df_for = df_for.set_index("time")

# concat metars an forecast
df_res = pd.concat([df_for,metars["brfg_o_l"]], axis = 1)
df_res_dropna = df_res.dropna()

#Heidke skill score ml
cm_ml = pd.crosstab(df_res.dropna().brfg_o_l, df_res.dropna().brfg_ml, margins=True,)
HSS_ml = 0
if cm_ml.shape == (3,3):# complete confusion matrix to calculate HSS
  a = cm_ml.values[0,0]
  b = cm_ml.values[1,0]
  c = cm_ml.values[0,1]
  d = cm_ml.values[1,1]
  HSS_ml = round(2*(a*d-b*c)/((a+c)*(c+d)+(a+b)*(b+d)),2)


#show results
st.markdown("**BR or FG**")
st.markdown("Reference (48 hours) Heidke skill score machine learning: 0.64")
st.markdown("Confusion matrix")
st.write(cm_ml)

fig, ax = plt.subplots(figsize=(10,6))
plt.plot(df_res_dropna.index, df_res_dropna['brfg_ml'],marker="^", markersize=8, 
         markerfacecolor='w', linestyle='');
plt.plot(df_res_dropna.index, df_res_dropna['brfg_o_l'],marker="*",markersize=8, 
         markerfacecolor='w', linestyle='');
plt.legend(('brfg ml', 'brfg observed'),)
plt.grid(True)
plt.title("Heidke skill score machine learning: {} ".format(HSS_ml))
st.pyplot(fig)


fig, ax = plt.subplots(figsize=(10,6))
plt.plot(df_for.index, df_for['brfg_ml'],marker="^",linestyle='');
plt.title("Forecast machine learning")
plt.grid(True)
st.pyplot(fig)

#show probabilistic results
prob = (np.concatenate((alg["pipe"].predict_proba(model_x_var),alg1["pipe"].predict_proba(model_x_var1)),axis =0)).transpose()
df_prob = (pd.DataFrame(prob,index =alg["pipe"].classes_ ).T.set_index(meteo_model[:48].index))
df_prob["time"] = meteo_model[:48].index
st.write("""Probabilistic results""")
AgGrid(round(df_prob,2)) 



#@title Precipitation
#open algorithm prec d0 d1
alg = pickle.load(open("algorithms/prec_LEVX_1km_time_d0.al","rb"))
alg1 = pickle.load(open("algorithms/prec_LEVX_1km_time_d1.al","rb"))

#select model variables
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]

# forecat prec from ml
prec_ml = alg["pipe"].predict(model_x_var)
prec_ml1 = alg1["pipe"].predict(model_x_var1)

#label metars prec data
metars["prec_o_l"] = "No RA/DZ"
mask = metars['wxcodes_o'].str.contains("RA")
metars.loc[mask,["prec_o_l"]] = "RA/DZ"
mask = metars['wxcodes_o'].str.contains("DZ")
metars.loc[mask,["prec_o_l"]] = "RA/DZ"

#label meteorological model prec0
prec0_l= ["RA/DZ" if c>0 else "No RA/DZ" for c in np.concatenate((model_x_var["prec0"],model_x_var1["prec0"]), axis=0)]

#set up dataframe forecast machine learning 
df_for = pd.DataFrame({"time":meteo_model[:48].index,
                       "prec_WRF": prec0_l,
                       "prec_ml": np.concatenate((prec_ml,prec_ml1),axis =0),})
df_for = df_for.set_index("time")

# concat metars an forecast
df_res = pd.concat([df_for,metars["prec_o_l"]], axis = 1)
df_res_dropna = df_res.dropna()

#Heidke skill score ml
cm_ml = pd.crosstab(df_res.dropna().prec_o_l, df_res.dropna().prec_ml, margins=True,)
HSS_ml = 0
if cm_ml.shape == (3,3):# complete confusion matrix to calculate HSS
  a = cm_ml.values[0,0]
  b = cm_ml.values[1,0]
  c = cm_ml.values[0,1]
  d = cm_ml.values[1,1]
  HSS_ml = round(2*(a*d-b*c)/((a+c)*(c+d)+(a+b)*(b+d)),2)

#Heidke skill score meteorological model
cm_wrf = pd.crosstab(df_res.dropna().prec_o_l, df_res.dropna().prec_WRF, margins=True,)
HSS_wrf = 0
if cm_wrf.shape == (3,3):# complete confusion matrix to calculate HSS
  a = cm_wrf.values[0,0]
  b = cm_wrf.values[1,0]
  c = cm_wrf.values[0,1]
  d = cm_wrf.values[1,1]
  HSS_wrf = round(2*(a*d-b*c)/((a+c)*(c+d)+(a+b)*(b+d)),2)

#show results
st.markdown("**Precipitation**")
st.markdown("Reference (48 hours) Heidke skill score meteorological model: 0.43")
st.markdown("Reference (48 hours) Heidke skill score machine learning: 0.55")
st.markdown("Confusion matrix machine learning")
st.write(cm_ml)
st.markdown("Confusion matrix meteorological model")
st.write(cm_wrf)

fig, ax = plt.subplots(figsize=(10,6))
plt.plot(df_res_dropna.index, df_res_dropna['prec_ml'],marker="^", markersize=8, 
         markerfacecolor='w', linestyle='');
plt.plot(df_res_dropna.index, df_res_dropna['prec_o_l'],marker="*",markersize=8, 
         markerfacecolor='w', linestyle='');
plt.plot(df_res_dropna.index, df_res_dropna['prec_WRF'],marker="v",markersize=8, 
         markerfacecolor='w', linestyle='');
plt.legend(('prec ml', 'prec observed',"precipitation WRF"),)
plt.grid(True)
plt.title("Heidke skill score meteorological model: {}\nHeidke skill score machine learning: {} ".format(HSS_wrf,HSS_ml))
st.pyplot(fig)


fig, ax = plt.subplots(figsize=(10,6))
plt.plot(df_for.index, df_for['prec_ml'],marker="^", markersize=8, markerfacecolor='w', linestyle='');
plt.plot(df_for.index, df_for['prec_WRF'],marker="v",markersize=8, markerfacecolor='w', linestyle='');
plt.legend(('prec ml', "precipitation WRF"),)
plt.title("Forecast machine learning versus WRF")
plt.grid(True)
st.pyplot(fig)

#show probabilistic results
prob = (np.concatenate((alg["pipe"].predict_proba(model_x_var),alg1["pipe"].predict_proba(model_x_var1)),axis =0)).transpose()
df_prob = (pd.DataFrame(prob,index =alg["pipe"].classes_ ).T.set_index(meteo_model[:48].index))
df_prob["time"] = meteo_model[:48].index
st.write("""Probabilistic results""")
AgGrid(round(df_prob,2)) 


#@title Cloud cover
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#open algorithm skyc1 d0 d1
alg = pickle.load(open("algorithms/skyc1_LEVX_1km_time_d0.al","rb"))
alg1 = pickle.load(open("algorithms/skyc1_LEVX_1km_time_d1.al","rb"))

#select model variables
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]

# forecat spd from ml
skyc1_ml = alg["pipe"].predict(model_x_var)
skyc1_ml1 = alg1["pipe"].predict(model_x_var1)

#set up dataframe forecast machine learning and WRF
df_for = pd.DataFrame({"time":meteo_model[:48].index,
                       "skyc1_ml": np.concatenate((skyc1_ml,skyc1_ml1),axis =0),})
df_for = df_for.set_index("time")

  
# concat metars an forecast
df_res = pd.concat([df_for,metars["skyc1_o"]],axis = 1)

#get accuracy
df_res_dropna = df_res.dropna()
acc_ml = round(accuracy_score(df_res_dropna.skyc1_o,df_res_dropna.skyc1_ml),2)

#show results
st.markdown("**Cloud cover level 1**")
st.markdown("Reference (48 hours) Accuracy machine learning: 0.64")

fig, ax = plt.subplots(figsize=(10,6))
plt.plot(df_res_dropna.index, df_res_dropna['skyc1_ml'], marker="^", markersize=8, 
         markerfacecolor='w', linestyle='')
plt.plot(df_res_dropna.index, df_res_dropna['skyc1_o'], marker="*", markersize=13,
         markerfacecolor='k', linestyle='');

plt.legend(('direction ml', 'direction observed'),)
plt.grid(True)
plt.title("Accuracy machine learning: {} ".format(acc_ml))
st.pyplot(fig)


fig, ax = plt.subplots(figsize=(10,6))
plt.plot(df_for.index, df_for['skyc1_ml'],marker="^",linestyle='');
plt.legend(('skyc1 ml'),)
plt.title("Forecast machine learning")
plt.grid(True)
st.pyplot(fig)

#show probabilistic results
prob = (np.concatenate((alg["pipe"].predict_proba(model_x_var),alg1["pipe"].predict_proba(model_x_var1)),axis =0)).transpose()
df_prob = (pd.DataFrame(prob,index =alg["pipe"].classes_ ).T.set_index(meteo_model[:48].index))

# Find the columns where all values are less than or equal to 5%
cols_to_drop = df_prob.columns[df_prob.apply(lambda x: x <= 0.05).all()]
df_prob.drop(cols_to_drop, axis=1, inplace=True)
df_prob["time"] = meteo_model[:48].index
st.write("""Probabilistic results""")
AgGrid(round(df_prob,2)) 

#@title Cloud height
#open algorithm  d0 d1
alg = pickle.load(open("algorithms/skyl1_LEVX_1km_time_d0.al","rb"))
alg1 = pickle.load(open("algorithms/skyl1_LEVX_1km_time_d1.al","rb"))

#select model variables
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]

# forecat skyl1 from ml
skyl1_ml = alg["pipe"].predict(model_x_var)
skyl1_ml1 = alg1["pipe"].predict(model_x_var1)

#label metars skyl1s data
#clouds height to feet or "M" no clooud
num = pd.to_numeric(metars.skyl1_o, errors="coerce")*1

#label more or less than 200 feet
interval = pd.IntervalIndex.from_tuples([(-1, 300),(300,7000)])
labels = ["<=300ft",">300ft"]
metars["skyl1_l"] = pd.cut(num, bins=interval,retbins=False,labels=labels)
metars["skyl1_l"] = metars["skyl1_l"].map({a:b for a,b in zip(interval,labels)})
metars["skyl1_l"] = metars["skyl1_l"].astype(str).replace("nan","No Cloud")


#set up dataframe forecast machine learning 
df_for = pd.DataFrame({"time":meteo_model[:48].index,
                       "skyl1_ml": np.concatenate((skyl1_ml,skyl1_ml1),axis =0),})
df_for = df_for.set_index("time")

# concat metars an forecast
df_res = pd.concat([df_for,metars["skyl1_l"]], axis = 1)
df_res_dropna = df_res.dropna()

#Accuracy score ml
cm_ml = pd.crosstab(df_res.dropna().skyl1_l, df_res.dropna().skyl1_ml, margins=True,)
acc_ml = round(accuracy_score(df_res_dropna.skyl1_l,df_res_dropna.skyl1_ml),2)


#show results
st.markdown("**Cloud height level 1**")
st.markdown("Reference (48 hours) Accuracy machine learning: 0.83")
st.markdown("Confusion matrix machine learning")
st.write(cm_ml)

fig, ax = plt.subplots(figsize=(10,6))
plt.plot(df_res_dropna.index, df_res_dropna['skyl1_ml'],marker="^", markersize=8, 
         markerfacecolor='w', linestyle='');
plt.plot(df_res_dropna.index, df_res_dropna['skyl1_l'],marker="*",markersize=8, 
         markerfacecolor='w', linestyle='');
plt.legend(('vis ml', 'vis observed'))
plt.grid(True)
plt.title("\nAccuracy machine learning: {} ".format(acc_ml))
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10,6))
plt.plot(df_for.index, df_for['skyl1_ml'],marker="^", markersize=8, 
         markerfacecolor='w', linestyle='');

plt.title("Forecast machine learning")
plt.grid(True)
st.pyplot(fig)

#show probabilistic results
prob = (np.concatenate((alg["pipe"].predict_proba(model_x_var),alg1["pipe"].predict_proba(model_x_var1)),axis =0)).transpose()
df_prob = (pd.DataFrame(prob,index =alg["pipe"].classes_ ).T.set_index(meteo_model[:48].index))
df_prob["time"] = meteo_model[:48].index
st.write("""Probabilistic results""")
AgGrid(round(df_prob,2))
