# -*- coding: utf-8 -*-
"""
The CESR_project is the main program for data analysis and visualization of PT VAINT.

The progam contains several additional modules:
    cosmo_data       ---> module for downloading and preparing COSMO data
    fluxnet_data     ---> module for downloading and preparing FLUXNET and EURONET data
    insitu_data      ---> module for downloading and preparing data from Linden and Lindenberg
    reanalysis_data  ---> module for downloading and preparing reanalysis data from E-OBS, HYRAS and GLEAM datasets
    system_operation ---> module with a system functions for cleaning data
    vis_module       ---> module for data visualization
    
    
Autors of project: Evgenii Churiulin, Merja Tölle, Center for Enviromental System
                                                   Research (CESR) 

                                                   
Acknowledgements: Vladimir Kopeikin, Denis Blinov



Current Code Owner: CESR, Evgenii Churiulin
phone:  +49  561 804-6142
fax:    +49  561 804-6116
email:  evgenychur@uni-kassel.de


History:
Version    Date       Name
---------- ---------- ----                                                   
    1.1    2021-04-15 Evgenii Churiulin, Center for Enviromental System Research (CESR)
           Initial release
                 

"""
# Import standart liblaries 
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import personal libraries
import vis_module       as vis                                                 
import reanalysis_data  as radata                                              
import fluxnet_data     as flnt                                                             
import cosmo_data       as csm_data                                            
import system_operation as stmo                                                
import insitu_data       as isd

# Improt methods for statistical analysis
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Start programm

# Special function for FLUXNET and EURONET data corrections
def get_ts(data_series, period, timestep, name, change_on):
    try:
        data = data_series[period].resample(timestep).mean() 
        return data
    except KeyError as error:   
        data = change_on    
        return data


# Calculations of GLEAM data with actual time step
#------------------------------------------------------------------------------
def get_gleam(df, param, step):
    '''
    Parameters
    ----------
    df : Series
        The actual parameter of GLEAM dataset for analysis.
    param : Objects
        The column name for actual GLEAM parameter.
    step : Objects
        The timestep for resampling of GLEAM data
    Returns
    -------
    df_final : DataFrame
        The dataframe with information about GLEAM data for each parameter
    '''
    gleam_list = []
    for  index in range(len(time_start)):   
        # Create time periods 
        daily_period  = pd.date_range(time_start[index],
                                      time_stop[index],
                                      freq = 'D')                              # dayly timesteps 
        #GLEAM data                
        gleam_list.append(df[daily_period].resample(step).mean())       
    # create new dataframe        
    df_new = pd.concat(gleam_list, axis = 0).reset_index()
    df_new['month'] = df_new['index'].dt.month
    df_new['day']   = df_new['index'].dt.day
    # final dataset
    df_final = df_new.groupby(['month','day']).agg({param  : 'mean'}).reset_index()
    
    # Create new index for time
    year = pd.Series(2013, index = df_final.index)
    month = df_final['month']
    day   = df_final['day']
    
    df_final['index_id'] = [pd.to_datetime('{}-{}-{}'.format(i, j, z), format='%Y-%m-%d') for i,j,z in zip(year, month,day)] 
    df_final = df_final.set_index(df_final['index_id']).drop(['index_id', 'month', 'day' ], axis = 1)  
    
    return df_final
#------------------------------------------------------------------------------

# Calculations annual mean values of COSMO-CLM parameters
#------------------------------------------------------------------------------
def get_cclm(df, params, time_start, time_stop, t_start, t_stop, step):
    '''
    Parameters
    ----------
    df : DataFrame
        The original COSMO-CLM data.
    params : List
        Names of COSMO-CLM parameters.
    time_start : timestemp
        The period for analisis (start)
    time_stop : timestemp
        The period for analisis (start)
    t_start : int
        first hour (start)
    t_stop : ind
        last hour  (stop)
    step : ind
        Timestep for resampling.
    Returns
    -------
    df_final : DataFrame
        The dataframe with information about COSMO-CLM parameters.
    '''
    df_list = []
    for  index in range(len(time_start)):   
        # Create time periods 
        hourly_period = pd.date_range(time_start[index], 
                                      time_stop[index], 
                                      freq = 'H')                              # hourly timesteps      
        # General time period for COSMO, FLUXNET and EURONET data
        res_period   = [x for x in hourly_period if x.hour >= t_start and x.hour <= t_stop]         
        # Create data
        df_cosmo  = csm_data.get_timeseries(df, params, res_period, 'D')              
        # Get COSMO values for 
        df_cosmo_mean = df_cosmo.resample(step).mean()
        # Get all data from period in one dataframe
        df_list.append(df_cosmo_mean)
    # create new dataframe        
    df_new = pd.concat(df_list, axis = 0).reset_index()
    df_new['month'] = df_new['Date'].dt.month
    df_new['day']   = df_new['Date'].dt.day
    
    
    df_final = df_new.groupby(['month','day']).agg({'AEVAP_S'  : 'mean',
                                                    'ALHFL_S'  : 'mean',
                                                    'ASHFL_S'  : 'mean',
                                                    'ZTRALEAV' : 'mean',
                                                    'ZVERBO'   : 'mean',
                                                    'RSTOM'    : 'mean',
                                                    'T_2M'     : 'mean',
                                                    'T_S'      : 'mean',
                                                    'TMAX_2M'  : 'mean',
                                                    'TMIN_2M'  : 'mean',
                                                    'TOT_PREC' : 'mean',
                                                    'W_SO'     : 'mean'}).reset_index() 
   
    # Create new index for time
    year = pd.Series(2013, index = df_final.index)
    month = df_final['month']
    day   = df_final['day']
    df_final['index_id'] = [pd.to_datetime('{}-{}-{}'.format(i, j, z), format='%Y-%m-%d') for i,j,z in zip(year, month,day)] 
    df_final = df_final.set_index(df_final['index_id']).drop(['index_id', 'month', 'day'], axis = 1)        
    return df_final
#------------------------------------------------------------------------------

# Calculating accumulated values of FLUXNET and EURONET data
#------------------------------------------------------------------------------
def get_flx_euro(df, params, time_start, time_stop, t_start, t_stop, step):

    fl_list = []
    for  index in range(len(time_start)):   
        # Create time periods 
        hourly_period = pd.date_range(time_start[index], time_stop[index], freq = 'H')  # hourly timesteps  
        # General time period for COSMO, FLUXNET and EURONET data
        res_period   = [x for x in hourly_period if x.hour >= t_start and x.hour <= t_stop]
        # Create a nan timeseries for changing incorrect values on NaN
        s_zero = pd.Series(np.nan, index = res_period)
        #FLUXNET data    
        data = get_ts(df, res_period, step, 'T2m FLUXNET', s_zero)  
        fl_list.append(data.dropna())
    df_new = pd.concat(fl_list, axis = 0).reset_index()  
    df_new.columns = ['Date', params]  
    df_new['month'] = df_new['Date'].dt.month
    df_new['day']   = df_new['Date'].dt.day
      
    df_final = df_new.groupby(['month','day']).agg({params  : 'mean'}).reset_index() 
    # Create new index for time
    year = pd.Series(2013, index = df_final.index)
    month = df_final['month']
    day   = df_final['day']
    # Create and set new indexes for data 
    df_final['index_id'] = [pd.to_datetime('{}-{}-{}'.format(i, j, z), format='%Y-%m-%d') for i,j,z in zip(year, month,day)] 
    df_final = df_final.set_index(df_final['index_id']).drop(['index_id', 'month', 'day'], axis = 1)   
    return df_final
#------------------------------------------------------------------------------


def corr_mae_rmse(df_mod, df_obs1, exp_name):
    stat_by_exper = {# Calculations experiments parameters
                     'mean'       : df_mod.mean(),
                     'std'        : df_mod.std(),
                     # Calculations statistics for OBS data 1
                     'corr v3.5a' : df_obs1.corr(df_mod),
                     'mae v3.5a'  : mean_absolute_error(df_obs1, df_mod),
                     'rmse v3.5a' : mean_squared_error(df_obs1 , df_mod)}
    
    df_stat = pd.DataFrame(data = stat_by_exper, index = [exp_name])
    return df_stat
#------------------------------------------------------------------------------


def corr_mae_rmse2(df_mod, df_obs1, df_obs2, exp_name):
    stat_by_exper = {# Calculations experiments parameters
                     'mean'       : df_mod.mean(),
                     'std'        : df_mod.std(),
                     # Calculations statistics for OBS data 1
                     'corr v3.5a' : df_mod.corr(df_obs1),
                     'mae v3.5a'  : mean_absolute_error(df_mod, df_obs1 ),
                     'rmse v3.5a' : mean_squared_error(df_mod , df_obs1 ),
                     # Calculations statistics for OBS data 2
                     'corr v3.5b' : df_mod.corr(df_obs1),
                     'mae v3.5b'  : mean_absolute_error(df_mod, df_obs2 ),
                     'rmse v3.5b' : mean_squared_error(df_mod , df_obs2 )}
    
    df_stat = pd.DataFrame(data = stat_by_exper, index = [exp_name])
    return df_stat
#------------------------------------------------------------------------------



# Get statistical parameters for one parameter for 4 experiments
#------------------------------------------------------------------------------
def stat_param(df1, df2, df3, df4, param, df_obs1):
    '''
    Parameters
    ----------
    df1 : DataFrame
        The CCLMref experiment.
    df2 : DataFrame
        The CCLMv3.5 experiment.
    df3 : DataFrame
        The CCLMv4.5 experiment.
    df4 : DataFrame
        The CCLMv4.5e experiment.
    param : Objects
        Parameter for statistical analysis.
    df_obs1 : DataFrame
        The dataframe with observational data HYRAS, E-OBS, FLUXNET, EURONET.

    Returns
    -------
    stat : DataFrame
        Dataset with statistical results.

    '''
    stat_cclm_ref  = corr_mae_rmse(df1[param], df_obs1, 'CCLMref'  )
    stat_cclm_v35  = corr_mae_rmse(df2[param], df_obs1, 'CCLMv3.5' )
    stat_cclm_v45  = corr_mae_rmse(df3[param], df_obs1, 'CCLMv4.5' )
    stat_cclm_v45e = corr_mae_rmse(df4[param], df_obs1, 'CCLMv4.5e')
    
    stat           = pd.concat([stat_cclm_ref, stat_cclm_v35 ,
                                stat_cclm_v45, stat_cclm_v45e], axis = 0 ).T   
    return stat
#------------------------------------------------------------------------------

# Get statistical parameters for one parameter for 4 experiments - 2 different dataset
#------------------------------------------------------------------------------
def stat_param2(df1, df2, df3, df4, param, df_obs1, df_obs2):
    '''
    Parameters
    ----------
    df1 : DataFrame
        The CCLMref experiment.
    df2 : DataFrame
        The CCLMv3.5 experiment.
    df3 : DataFrame
        The CCLMv4.5 experiment.
    df4 : DataFrame
        The CCLMv4.5e experiment.
    param : Objects
        Parameter for statistical analysis.
    df_obs1 : DataFrame
        The dataframe with observational data GLEAM v3.5a
    df_obs2 : DataFrame
        The dataframe with observational data GLEAM v3.5b
    Returns
    -------
    stat : DataFrame
        Dataset with statistical results.
    '''
    stat_cclm_ref  = corr_mae_rmse2(df1[param], df_obs1, df_obs2, 'CCLMref'  )
    stat_cclm_v35  = corr_mae_rmse2(df2[param], df_obs1, df_obs2, 'CCLMv3.5' )
    stat_cclm_v45  = corr_mae_rmse2(df3[param], df_obs1, df_obs2, 'CCLMv4.5' )
    stat_cclm_v45e = corr_mae_rmse2(df4[param], df_obs1, df_obs2, 'CCLMv4.5e') 
        
    stat = pd.concat([stat_cclm_ref, stat_cclm_v35 ,
                      stat_cclm_v45, stat_cclm_v45e], axis = 0 ).T   
    return stat
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Section for logical data types ----> Don't change them
#------------------------------------------------------------------------------
ldaily   = True                                                                # hour  data timestep                                                      
lperiod  = True                                                                # daily data timestep                                                          
llperiod = True                                                                # user  data timestep                                                                                                               

#------------------------------------------------------------------------------
# Section: Constant parameters
#------------------------------------------------------------------------------
t0melt = 273.15                                                                

#------------------------------------------------------------------------------
# Section for users: Parameters can be changed by user
#------------------------------------------------------------------------------                                  

#------------------------------------------------------------------------------
# Parameters for: period_cal    --> timestep of data (1 - hourly, 
#                                                     2 - Montly,
#                                                     3 - User interval)
#
#    
#             input_region      --> (P - park,
#                                    L - linden,
#                                    Li - lindenberg)
# 
#             input_station     --> (For Park region --> RuR --> 'Rollesbroich      (lat - 50.62; lon - 6.30; 
#                                                                                          Land type - Grass)
#                                                        RuS --> 'Selhausen Juelich (lat - 50.86; lon - 6.44;
#                                                                                          Land type - Crops) 
#                                                        SeH --> 'Selhausen         (lat - 50.87; lon - 6.44;
#                                                                                        Land type - Agricultural area)
#                                    For Linden      --> LiN --> 'Linden'           (lat - ; lon - ; Land type - Grass)
#                                    For Lindenberg  --> LiD --> 'Lindenberg'       (lat - ; lon - ; Land type - ))
#           
#             time_array        --> time step for resample initial dataframe
#                                     period_cal = 1 then time_array = 'H'
#                                     period_cal = 2 then time_array = 'D'
#                                     period_cal = 3 then time_array = '2D'   
#------------------------------------------------------------------------------
period_cal    = '3'               
input_region  = 'P'
input_station = 'RuR'           
time_array   = ['H', 'D', 'D']

#------------------------------------------------------------------------------
# Define paths for data
#------------------------------------------------------------------------------    

mf_com    = 'C:/Users/Churiulin/Desktop/COSMO_RESULTS/'                        # The main paths which are general for all data

path_exit = mf_com + 'Python_results/'                                         # The path for results

 
if input_region == 'P':                                                        # Special parameters for Parc domain
    print('The Parc domain was chosen: \n')     
    sf_region = 'PARC/'                                                        # folder with data for Parc domain  
    fn_region = 'parc'                                                         # filename part 
               
elif input_region == 'L':                                                      # Special parameters for Linden domain
    print('The Linden domain was chosen: \n')
    sf_region = 'LINDEN/'                                                     
    fn_region = 'linden'                                                  
    
elif input_region == 'Li':                                                     # Special parameters for Lindenberg domain
    print('The Lindenberg domain was chosen: \n')
    sf_region = 'LINDENBERG/'                                                 
    fn_region = 'lindenberg'                                                
else:
    print ('Error: No data for domain --> line 161')


sf_cclm_ref   = mf_com + 'COSMO/' + sf_region + 'CTR/'                         # COSMO_CTR
sf_cclm_v35   = mf_com + 'COSMO/' + sf_region + 'v3.5/'                        # COSMO_v3.5
sf_cclm_v45   = mf_com + 'COSMO/' + sf_region + 'v4.5/'                        # COSMO_v4.5
sf_cclm_v45e  = mf_com + 'COSMO/' + sf_region + 'v4.5e/'                       # COSMO_v4.5e

sf_fluxnet    = mf_com + 'IN-SITU/FLUXNET/'                                    # FLUXNET data    
sf_euronet    = mf_com + 'IN-SITU/EURONET/'                                    # EURONET data
sf_linden     = mf_com + 'IN-SITU/LINDEN/'                                     # Linden  data 
sf_lindenberg = mf_com + 'IN-SITU/LINDENBERG/'                                 # Lindenberg  data 

sf_eobs       = mf_com +'REANALYSIS/EOBS/'                                     # EOBS data
sf_hyras      = mf_com +'REANALYSIS/HYRAS/'                                    # HYRAS data
sf_gleam      = mf_com +'REANALYSIS/GLEAM/'                                    # GLEAM data  

fn_cosmo      = '_ts_mean_1999_2015.csv'                                       # General part in file name for COSMO
                                                                               # and experiments

#------------------------------------------------------------------------------ 
# Names of parameters for COSMO data
#------------------------------------------------------------------------------
clm_name = ['AEVAP_S'  , 'ALHFL_BS' , 'ALHFL_PL', 'ALHFL_S' , 'ASHFL_S', 
            'ZTRALEAV' , 'ZTRANG'   , 'ZTRANGS' , 'ZVERBO'  , 'QV_2M'  , 
            'QV_S'     , 'RELHUM_2M', 'RSTOM'   , 'T_2M'    , 'T_S'    ,
            'TMAX_2M'  , 'TMIN_2M'  , 'PS'      , 'TOT_PREC', 'W_SO'   ]



#------------------------------------------------------------------------------
# Names of plot labels
#------------------------------------------------------------------------------
name_1 = ['Amount of water evaporation (surface) - AEVAP_S'                ,
          'Average latent heat flux from bare soil evaporation - ALHFL_BS' ,
          'Average latent heat flux from plants - ALHFL_PL'                ,
          'Average latent heat flux (surface) - ALHFL_S'                   ,
          'Average sensible heat flux (surface) - ASHFL_S'                 ,
          'Transpiration rate of dry leaves - ZTRALEAV'                    ,
          'Transpiration contribution by the first layer - ZTRANG'         ,
          'Total transpiration - ZTRANGS'                                  ,
          'Total evapotranspiration - ZVERBO'                              , 
          'Specific_humidity in 2m - QV_2M'                                ,
          'Surface_specific_humidity - QV_S'                               ,
          'Relative_humidity  in 2m - RELHUM_2M'                           ,
          'Stomata resistance - RSTOM'                                     , 
          'Air temperature  in 2m - T_2M'                                  ,
          'Soil temperature  in 2m - T_S'                                  , 
          'Maximum air temperature in 2m - Tmax'                           ,
          'Minimum air temperature - 2m - Tmin'                            ,          
          'Surface pressure - PS'                                          ,
          'Total precipitation - TOT_PREC'                                 ,
          'Soil moisture - W_SO'                                           ]         
          
#------------------------------------------------------------------------------
# Names of y axis
#------------------------------------------------------------------------------                                                  
y_label   = ['AEVAP\u209B, kg m \u207b\u00B2'     , 'ALHFL_BS, W m \u207b\u00B2'        ,
             'ALHFL\u209A\u2097, W m \u207b\u00B2', 'ALHFL\u209B, W m \u207b\u00B2'     , 
             'ASHFL\u209B, W m \u207b\u00B2'      , 'ZTRALEAV, mm day \u207b\u00B9'     ,
             'ZTRANG, mm day \u207b\u00B9'        , 'ZTRANGS, mm day \u207b\u00B9'      ,
             'ZVERBO, mm day \u207b\u00B9'        , 'QV\u2082\u2098, kg kg \u207b\u00B9',
             'QV\u209B, kg kg \u207b\u00B9'       , 'RH\u2082\u2098, %'                 ,
             'RSTOM, s m \u207b\u00B9'            , 'T\u2082\u2098, C \u2070'           ,
             'T\u209B, C \u2070'                  , 'Tmax\u2082\u2098, C\u2070'         ,
             'Tmin\u2082\u2098, C\u2070'          , 'PS, hPa'                           ,
             'TOT_PREC, mm'                       , 'W_SO'                              ] 
                       
#------------------------------------------------------------------------------
# Select actual limits for plots arrording to timestep   
#------------------------------------------------------------------------------
if period_cal == '1':                                                          # DAILY data with hourly timesteps       
    print(f'Data format - daily data, Timestep - {time_array[0]}')       
    # Logical types for data
    daily_time_step   = True                                                   
    monthly_time_step = False     
    long_time_step    = False  
    # Time step
    step4resample                = time_array[0]  
    # Lower limit
    y_min  = [  -0.05,    -5.0,    -5.0,    -25.0   ,    0.0     ,     0.0,    # AEVAP, ALHFL_BS, ALHFL_PL, ALHFL_S, ASHFL_S, AZTRALEAV,
                 0.0 ,     0.0,     0.0,      0.0   ,   -2.0     ,    30.0,    # AZTRANG, AZTRANGS,  AZVERBO,   QV_2M,    QV_S, RELHUM_2M,  
                 0.0 ,   -15.0,   -15.0,    -15.0   ,    15.0    ,   900.0]    # RSTOM  ,     T_2M,      T_S, Tmax, Tmin      PS 
    # Upper limit
    y_max  = [   0.41,    50.1,    50.1,    250.1   ,  250.1     ,    50.1,    
                50.1 ,    50.1,    50.0,      0.0151,    2.01    ,   100.1,     
              5001.0 ,    35.1,    35.0,     35.0   ,   35.1     ,  1050.1]    
    # Step
    y_step = [   0.05,     5.0,     5.0,     25.0   ,   25.0     ,    10.0,    
                10.0 ,    10.0,    10.0,      0.005 ,    0.5     ,    10.0,    
               500.0 ,    10.0,    10.0,     10.0   ,    10.0    ,    25.0]    

elif period_cal == '2':                                                        # MONTLY data with daily timesteps    
    print(f'Data format - daily data, Timestep - {time_array[1]}')
    # Logical types for data
    daily_time_step   = False                                                  
    monthly_time_step = True 
    long_time_step    = False
    # Time step
    step4resample                = time_array[1]        
    # Lower limit 
    y_min = [    0.0 ,  -25.0,   -25.0,    -25.0   ,    0.0     ,     0.0,     # AEVAP  , ALHFL_BS, ALHFL_PL, ALHFL_S, ASHFL_S, AZTRALEAV,
                 0.0 ,    0.0,     0.0,      0.0   ,    0.0     ,    30.0,     # AZTRANG, AZTRANGS,  AZVERBO,   QV_2M,    QV_S, RELHUM_2M, 
                 0.0 ,  -15.0,   -15.0,    900.0                         ]     # RSTOM  ,     T_2M,      T_S,      PS 
    # Upper limit
    y_max = [    6.01,  125.1,   125.1,    125.1   ,    250.1   ,    50.1,     
                50.1 ,   50.1,    50.1,      0.0151,      0.0151,   100.1,  
             20000.1 ,   35.1,    35.0,   1050.1                         ]  
    # Step
    y_step = [   0.50,   25.0,    25.0,     25.0   ,     25.0   ,    10.0,     
                10.0 ,   10.0,    10.0,      0.005 ,      0.005 ,    10.0,      
              5000.0 ,   10.0,    10.0,     25.0                         ]     

elif period_cal == '3':                                                        # USER data with user timesteps  
    print(f'Data format - daily data, Timestep - {time_array[2]}') 
    # Logical types for data
    daily_time_step   = False                                                  
    monthly_time_step = False    
    long_time_step    = True 
    # Time step
    step4resample = time_array[2]
    # Lower limit
    y_min = [    0.0 ,    0.0,     0.0,      0.0   ,    -50.0   ,     0.0,     # AEVAP  , ALHFL_BS, ALHFL_PL, ALHFL_S, ASHFL_S, ZTRALEAV, 
                 0.0 ,    0.0,     0.0,      0.0   ,      0.0   ,    40.0,     # ZTRANG, ZTRANGS,  ZVERBO,   QV_2M,    QV_S, RELHUM_2M,
                 0.0 ,    0.0,     0.0,      0.0   ,      0.0    ,  975.0,     # RSTOM  ,     T_2M,      T_S,  Tmax, Tmin
                 0.0 ,    0.02]                                                #  PS, TOT_PREC, W_SO     
    # Upper limit
    y_max = [    4.5 ,   75.1,    30.1,    125.1   ,    100.1   ,     4.1,           
                 1.51,    4.1,     6.1,      0.0151,      0.0151,   100.1,       
              20000.1,   30.1,    30.1,     30.1   ,     30.1   ,  1025.1,
              20.0   ,     0.08]                                 
    # Step
    y_step = [   0.5 ,   15.0,     5.0,     25.0   ,     25.0   ,     0.5,           
                 0.25,    0.5,     0.5,      0.005 ,      0.005 ,    10.0,      
              2000.0 ,    5.0,     5.0,      5.0   ,      5.0   ,    10.0,
              5.0    ,    0.01]                          
else:
    print ('Error: Incorrect actual period!')
    sys.exit()      

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Section: Clear the privios results
#------------------------------------------------------------------------------
while True:
    input_control = input('Do you want to remote previous data: yes - y; no - n \n') 
    if input_control == 'y':
        clean_result = stmo.dep_clean(path_exit)
        print ('All previous results were deleted: \n')        
        break
    
    elif input_control == 'n': 
        print ('All previous results were saved and ' +
               'in case of the same name - data will be rewritten: \n')
        break
    else:
        print ('Error: Incorrect format, repeat! \n')

#------------------------------------------------------------------------------


#==============================================================================
# Get initial data: COSMO, FLUXNET, EURONET, GLEAM, HYRAS, E-OBS
#==============================================================================

# The COSMO data has a hourly timestep
df_cclm_ref  = csm_data.cosmo_data(sf_cclm_ref , fn_cosmo, clm_name)
df_cclm_v35  = csm_data.cosmo_data(sf_cclm_v35 , fn_cosmo, clm_name)
df_cclm_v45  = csm_data.cosmo_data(sf_cclm_v45 , fn_cosmo, clm_name)
df_cclm_v45e = csm_data.cosmo_data(sf_cclm_v45e, fn_cosmo, clm_name)

# The FLUXNET and EURONET has a hourly timestep
df_fluxnet, station_name_plot = flnt.fluxnet_data(sf_fluxnet, input_station)   # get FLUXNET data
df_euronet                    = flnt.euronet_data(sf_euronet, input_station)   # get EORONET data      
       
# The GLEAM, E-OBS and HYRAS datasets has a daily timestep
df_eobs          = radata.eobs_data(sf_eobs, fn_region)                        # get E-OBS data
df_hyras         = radata.hyras_data(sf_hyras, fn_region)                      # get HYRAS data
df_v35a, df_v35b = radata.gleam_data(sf_gleam, fn_region)                      # get GLEAM data

#==============================================================================
# Get time periods
#==============================================================================
'''
# Example for several periods of time
time_id_1 = pd.to_datetime(['2010-06-01'   , '2013-06-01'   ])
time_id_2 = pd.to_datetime(['2010-08-31 23', '2013-08-31 23'])

# Additional examples for work with time
houry_time_period = pd.date_range(start = '2010-06-01', end = '2010-08-31 23', freq = 'H')
dayly_time_period = pd.date_range(start = '2010-06-01', end = '2010-08-31 23', freq = 'D')
'''

time_start = pd.to_datetime(['2010-05-01'   , '2011-05-01'   , 
                             '2012-05-01'   , '2013-05-01'   , 
                             '2014-05-01'   , '2015-05-01'   ])
time_stop  = pd.to_datetime(['2010-09-01 23', '2011-09-01 23', 
                             '2012-09-01 23', '2013-09-01 23', 
                             '2014-09-01 23', '2015-09-01 23'])

#time_start = pd.to_datetime(['2013-07-01'   ])
#time_stop  = pd.to_datetime(['2013-09-01 23'])
t_start = 00                                                                    # parameter --> from  
t_stop  = 23                                                                   # parameter --> to



#==============================================================================
# Section: Get annual data over the period
#==============================================================================
# Annual COSMO-CLM data      
days = 7

cclm_ref  = get_cclm(df_cclm_ref , clm_name, time_start   ,                    # Reference experiment
                                             time_stop    , 
                                             t_start      ,  
                                             t_stop       ,
                                             step4resample).apply(lambda x: x.rolling(days).mean().dropna())

cclm_v35  = get_cclm(df_cclm_v35 , clm_name, time_start   ,                    # CCLMv3.5 experiment
                                             time_stop    , 
                                             t_start      ,
                                             t_stop       ,
                                             step4resample).apply(lambda x: x.rolling(days).mean().dropna())

cclm_v45  = get_cclm(df_cclm_v45 , clm_name, time_start   ,                    # CCLMv4.5 experiment
                                             time_stop    , 
                                             t_start      ,
                                             t_stop       , 
                                             step4resample).apply(lambda x: x.rolling(days).mean().dropna())

cclm_v45e = get_cclm(df_cclm_v45e, clm_name, time_start   ,                    # CCLMv4.5e experiment
                                             time_stop    , 
                                             t_start      , 
                                             t_stop       ,
                                             step4resample).apply(lambda x: x.rolling(days).mean().dropna())

# Annual GLEAM data (v3.5a)    
ep_v35a = get_gleam(df_v35a['Ep'], 'Ep', step4resample).apply(lambda x: x.rolling(days).mean().dropna())
et_v35a = get_gleam(df_v35a['Et'], 'Et', step4resample).apply(lambda x: x.rolling(days).mean().dropna())

# Annual GLEAM data (v3.5b)
ep_v35b = get_gleam(df_v35b['Ep'], 'Ep', step4resample).apply(lambda x: x.rolling(days).mean().dropna()) 
et_v35b = get_gleam(df_v35b['Et'], 'Et', step4resample).apply(lambda x: x.rolling(days).mean().dropna())


# Annual HYRAS and E-OBS data
t2m_hyras  = get_gleam(df_hyras['T_2M'] , 'T_2M' , step4resample).apply(lambda x: x.rolling(days).mean().dropna())
tmax_hyras = get_gleam(df_hyras['T_MAX'], 'T_MAX', step4resample).apply(lambda x: x.rolling(days).mean().dropna())
tmin_hyras = get_gleam(df_hyras['T_MIN'], 'T_MIN', step4resample).apply(lambda x: x.rolling(days).mean().dropna())   
ts_hyras   = get_gleam(df_hyras['T_S']  , 'T_S'  , step4resample).apply(lambda x: x.rolling(days).mean().dropna())
   
# Annual FLUXNET data
t2m_flux = get_flx_euro(df_fluxnet['T2m'], 'T2m', time_start   , 
                                                  time_stop    ,
                                                  t_start      , 
                                                  t_stop       ,
                                                  step4resample).apply(lambda x: x.rolling(days).mean().dropna())

le_flux  = get_flx_euro(df_fluxnet['LE'] , 'LE' , time_start   ,
                                                  time_stop    , 
                                                  t_start      , 
                                                  t_stop       , 
                                                  step4resample).apply(lambda x: x.rolling(days).mean().dropna())
# Annual EURONET data
t2m_euro = get_flx_euro(df_euronet['TA'] , 'TA' , time_start   ,
                                                  time_stop    ,
                                                  t_start      ,
                                                  t_stop       ,
                                                  step4resample).apply(lambda x: x.rolling(days).mean().dropna())

le_euro  = get_flx_euro(df_euronet['LE'] , 'LE' , time_start   , 
                                                  time_stop    ,
                                                  t_start      ,
                                                  t_stop       ,
                                                  step4resample).apply(lambda x: x.rolling(days).mean().dropna())

h_euro   = get_flx_euro(df_euronet['H']  , 'H'  , time_start   , 
                                                  time_stop    ,
                                                  t_start      ,
                                                  t_stop       ,
                                                  step4resample).apply(lambda x: x.rolling(days).mean().dropna())

#------------------------------------------------------------------------------
# Section: Visualisation of annual data
#------------------------------------------------------------------------------

# Additional information for plot title: General            

clm_params = ['AEVAP_S', 'ZVERBO'  , 'ALHFL_S', 'ASHFL_S',
              'T_2M'   , 'TMAX_2M' , 'TMIN_2M', 'T_S'    ,
              'W_SO'   , 'TOT_PREC']

clm_ylabel = ['AEVAP\u209B, kg m \u207b\u00B2',
              'ZVERBO, mm day \u207b\u00B9'   ,
              'ALHFL\u209B, W m \u207b\u00B2' ,
              'ASHFL\u209B, W m \u207b\u00B2' ,
              'T\u2082\u2098, C \u2070'       ,
              'Tmax\u2082\u2098, C\u2070'     ,
              'Tmin\u2082\u2098, C\u2070'     ,
              'T\u209B, C \u2070'             ,
              'W_SO'                          ,    
              'TOT_PREC, mm'                  ]


# 2010 - 2015
y_min  = [ 1.5 , 1.5 ,  30.0,    0.0, 10.0, 10.0,  5.0,  10.0, 0.02 ,   0.0]
y_max  = [ 4.01, 4.51, 105.1,   75.1, 30.1, 30.1, 20.1,  30.1, 0.081,  10.0]     
y_step = [ 0.5 , 0.5 ,  15.0,   15.0,  5.0,  5.0,  5.0,   5.0, 0.01 ,   2.0]
    
# 2013 year
#y_min  = [ 0.0 ,  0.0,   0.0,    0.0, 15.0, 15.0,  5.0,  15.0, 0.02 ,   0.0]
#y_max  = [ 1.01, 16.1, 300.1,  350.1, 35.1, 35.1, 20.1,  35.1, 0.081,  10.0]     
#y_step = [ 0.25,  4.0,  50.0,   50.0,  5.0,  5.0,  5.0,   5.0, 0.01 ,   2.0]        
 
       
for i, pars in enumerate(clm_params):
    # Define in-situ data type
    # 2013   - ep, et
    if pars == 'AEVAP_S':
        gleam_v35a = et_v35a['Et']
        gleam_v35b = et_v35b['Et']
    elif pars == 'ZVERBO':
        gleam_v35a = ep_v35a['Ep']
        gleam_v35b = ep_v35b['Ep']
    elif pars == 'ALHFL_S':
        heat_flux = le_flux['LE']
    elif pars == 'ASHFL_S':
        heat_flux = h_euro['H']
    elif pars == 'T_2M':
        hyras_temp = t2m_hyras['T_2M']
    elif pars == 'TMAX_2M':
        hyras_temp = tmax_hyras['T_MAX']
    elif pars == 'TMIN_2M':
        hyras_temp = tmin_hyras['T_MIN']
    else:        
        hyras_temp = ts_hyras['T_S']
               
    # Create area for plot
    fig = plt.figure(figsize = (12,7))
    ax  = fig.add_subplot(111) 

    #dplot = vis.plots4(ax, cclm_ref[pars], cclm_v35[pars] ,                          
    #                       cclm_v45[pars], cclm_v45e[pars],
    #                       'CCLMref'     , 'CCLMv3.5'     , 
    #                       'CCLMv4.5'    , 'CCLMv4.5e'    )
    # Settings for plots          
    #dplot = vis.lplots_stomata2(ax, clm_ylabel[i], 'upper right' ,
    #                                    y_min[i], y_max[i], y_step[i])

    
    if pars in ('AEVAP_S', 'ZVERBO' ):
        # Create plots for AEVAP and ZVERBO
        dplot = vis.plots6(ax, cclm_ref[pars], cclm_v35[pars] ,                          
                               cclm_v45[pars], cclm_v45e[pars],
                               gleam_v35a    , gleam_v35b     ,           
                               'CCLMref'     , 'CCLMv3.5'     , 
                               'CCLMv4.5'    , 'CCLMv4.5e'    ,
                               'GLEAM 3.5a'  , 'GLEAM 3.5b'   )
        
        # Settings for plots          
        dplot = vis.lplots_stomata2(ax, clm_ylabel[i], 'upper left' ,
                                        y_min[i], y_max[i], y_step[i])
    
    elif pars in ('T_2M', 'TMAX_2M', 'TMIN_2M', 'T_S'):
        dplot = vis.plots5(ax, cclm_ref[pars], cclm_v35[pars] , 
                               cclm_v45[pars], cclm_v45e[pars], hyras_temp,                    
                               'CCLMref' , 'CCLMv3.5' , 
                               'CCLMv4.5', 'CCLMv4.5e', 'OBS')
        # Settings for plots          
        dplot = vis.lplots_stomata2(ax, clm_ylabel[i], 'upper left' ,
                                        y_min[i], y_max[i], y_step[i])          
    
    elif pars in ('ALHFL_S', 'ASHFL_S'):
        # Create plots for ALHFL_S and ASHFL_S
        dplot = vis.plots5(ax, cclm_ref[pars],  cclm_v35[pars],
                               cclm_v45[pars], cclm_v45e[pars], heat_flux,                                                       
                               'CCLMref' , 'CCLMv3.5', 
                               'CCLMv4.5', 'CCLMv4.5e', 'OBS' )  
        # Settings for plots          
        dplot = vis.lplots_stomata2(ax, clm_ylabel[i], 'upper right' , 
                                        y_min[i], y_max[i], y_step[i])    
    elif pars in ('W_SO'):
        # Create plots for ALHFL_S and ASHFL_S
        dplot = vis.plots4(ax, cclm_ref[pars],  cclm_v35[pars],
                               cclm_v45[pars], cclm_v45e[pars],                                                       
                               'CCLMref', 'CCLMv3.5', 'CCLMv4.5', 'CCLMv4.5e')  
        # Settings for plots          
        dplot = vis.lplots_stomata2(ax, clm_ylabel[i], 'upper right' , 
                                        y_min[i], y_max[i], y_step[i])            
    else:
        # Create plots for ALHFL_S and ASHFL_S
        dplot = vis.plots4_tot_prec(ax, cclm_ref[pars],  cclm_v35[pars],
                                        cclm_v45[pars], cclm_v45e[pars],                                                       
                                        'CCLMref', 'CCLMv3.5'  , 
                                        'CCLMv4.5', 'CCLMv4.5e')  
        # Settings for plots          
        dplot = vis.lplots_stomata2(ax, clm_ylabel[i], 'upper right' , 
                                        y_min[i], y_max[i], y_step[i])          
        
        
    # Create output plot name    
    output_name = f'{pars}.png'
    # Save plot            
    plt.savefig(path_exit + output_name, format = 'png', dpi = 300) 
    # Clean memory
    plt.close(fig)        
    plt.gcf().clear() 

 
#==============================================================================
# Section: Statistical analisys (Points)
#==============================================================================
# Statistics for GLEAM datasets - AEVAP
stat_aevap  = stat_param2(cclm_ref, cclm_v35, cclm_v45, cclm_v45e, 'AEVAP_S', et_v35a['Et'], et_v35b['Et'])
# Statistics for GLEAM datasets - ZVERBO
stat_zverbo = stat_param2(cclm_ref, cclm_v35, cclm_v45, cclm_v45e, 'ZVERBO' , ep_v35a['Ep'], ep_v35b['Ep'])
# Statistics for ALHFL_S 
stat_alhfl  = stat_param(cclm_ref, cclm_v35, cclm_v45, cclm_v45e, 'ALHFL_S', le_flux['LE'])
# Statistics for ASHFL_S 
stat_ashfl  = stat_param(cclm_ref, cclm_v35, cclm_v45, cclm_v45e, 'ASHFL_S', h_euro['H'])
# Statistics for T_2M
stat_t2m    = stat_param(cclm_ref, cclm_v35, cclm_v45, cclm_v45e, 'T_2M'   , t2m_hyras['T_2M'])
# Statistics for TMAX_2M
#stat_tmax   = stat_param(cclm_ref, cclm_v35, cclm_v45, cclm_v45e, 'TMAX_2M', tmax_hyras['T_MAX'])
# Statistics for TMIN_2M
#stat_tmin   = stat_param(cclm_ref, cclm_v35, cclm_v45, cclm_v45e, 'TMIN_2M', tmin_hyras['T_MIN'])
# Statistics for T_S
stat_ts     = stat_param(cclm_ref, cclm_v35, cclm_v45, cclm_v45e, 'T_S'    , ts_hyras['T_S'])





"""
#==============================================================================
# Old version 
#==============================================================================

for  index in range(len(time_start)):   
    #--------------------------------------------------------------------------
    # Create time periods 
    #-------------------------------------------------------------------------- 
    hourly_period = pd.date_range(time_start[index], time_stop[index], freq = 'H')  # hourly timesteps
    daily_period  = pd.date_range(time_start[index], time_stop[index], freq = 'D')  # dayly timesteps 
    
    # General time period for COSMO, FLUXNET and EURONET data
    res_period   = [x for x in hourly_period if x.hour >= t_start and x.hour <= t_stop]

    
    #--------------------------------------------------------------------------
    # Subsection: Create data for plots
    #--------------------------------------------------------------------------

    # COSMO data --> daily mean
    cclm_ref  = csm_data.get_timeseries(df_cclm_ref , clm_name, res_period, 'D')   # CCLMref  --> original COSMO       
    cclm_v35  = csm_data.get_timeseries(df_cclm_v35 , clm_name, res_period, 'D')   # CCLMv35  --> experiment    
    cclm_v45  = csm_data.get_timeseries(df_cclm_v45 , clm_name, res_period, 'D')   # CCLMv45  --> experiment           
    cclm_v45e = csm_data.get_timeseries(df_cclm_v45e, clm_name, res_period, 'D')   # CCLMv45e --> previous version parc_v45  
    
    # Get COSMO values for 
    cclm_mean      = cclm_ref.resample(step4resample).mean()
    cclm_v35_mean  = cclm_v35.resample(step4resample).mean()
    cclm_v45_mean  = cclm_v45.resample(step4resample).mean()
    cclm_v45e_mean = cclm_v45e.resample(step4resample).mean()

    s_zero = pd.Series(-1, index = res_period)
    #FLUXNET data    
    t2m_flux = get_ts(df_fluxnet['T2m'], res_period, step4resample, 'T2m FLUXNET', s_zero)  
    le_flux  = get_ts(df_fluxnet['LE'] , res_period, step4resample, 'LE FLUXNET' , s_zero) 
    ts_flux  = get_ts(df_fluxnet['Ts'] , res_period, step4resample, 'Ts FLUXNET' , s_zero) 
    pa_flux  = get_ts(df_fluxnet['Pa'] , res_period, step4resample, 'Pa FLUXNET' , s_zero) 

    #EURONET data
    t2m_euro = get_ts(df_euronet['TA'] , res_period, step4resample, 'TA EURONET' , s_zero)
    le_euro  = get_ts(df_euronet['LE'] , res_period, step4resample, 'LE EURONET' , s_zero)
    t_s_euro = get_ts(df_euronet['TS'] , res_period, step4resample, 'TS EURONET' , s_zero)    
    rh_euro  = get_ts(df_euronet['RH'] , res_period, step4resample, 'RH EURONET' , s_zero)
    h_euro   = get_ts(df_euronet['H']  , res_period, step4resample, 'H EURONET'  , s_zero)
    #pa_euro  = df_euronet['PA'][res_period].resample(step4resample).mean() * 10.0
      
    # HYRAS data
    t2m_hyras = df_hyras['T_2M'][daily_period].resample(step4resample).mean()        

    #E-OBS data
    t2m_eobs  = df_eobs['T_2M'][daily_period].resample(step4resample).mean()       
                   
    #GLEAM data v3.5a        
    ep_v35a = df_v35a['Ep'][daily_period].resample(step4resample).mean()                  
    et_v35a = df_v35a['Et'][daily_period].resample(step4resample).mean()                  
   
    #GLEAM data v3.5b 
    ep_v35b = df_v35b['Ep'][daily_period].resample(step4resample).mean()
    et_v35b = df_v35b['Et'][daily_period].resample(step4resample).mean()                  

    #--------------------------------------------------------------------------
    # Subsection: Data vizualization - Plots for all parameters   
    #--------------------------------------------------------------------------
   
    # Additional information for plot title: General            
    time_int_1 = str(time_start[index])[0:10]                                  # The date of period start --> need only for print
    time_int_2 =  str(time_stop[index])[0:10]                                  # The date of period stop  --> need only for print
    date_ind   = f'Time step: {time_int_1} to {time_int_2}'                    # The full date of period  --> need for plot label          
    name_3     = 'Hours'                                                       # x - label for hourly plots   
    l_p        = 'upper left'                                                  # The position of legend
    nst        = station_name_plot                                             # the name of the research station

    for k in range(len(clm_name)):       
        # Create area for plot
        fig = plt.figure(figsize = (14,10))
        ax  = fig.add_subplot(111)      
        
        if clm_name[k] in ('ALHFL_BS', 'ALHFL_PL' , 'ZTRALEAV', 'ZTRANG'   ,   # Plot data parameters
                           'ZTRANGS' , 'QV_2M'    , 'QV_S'    , 'RELHUM_2M', 
                           'RSTOM'   , 'PS'       , 'TOT_PREC', 'W_SO'     ):
            
            print (f'Plot - {clm_name[k]}, period: {time_int_1} to {time_int_2}')
                        
            plot4par = vis.plots4(ax, cclm_mean[clm_name[k]]    ,  cclm_v35_mean[clm_name[k]],
                                      cclm_v45_mean[clm_name[k]], cclm_v45e_mean[clm_name[k]],
                                      'CCLMref'   , 'CCLMv3.5'  , 'CCLMv4.5'    , 'CCLMv4.5e')
                                         
            if ldaily == daily_time_step:               
                plot4par = vis.dplots(ax, name_1[k], y_label[k], name_3  , date_ind ,  
                                          nst, l_p ,  y_min[k], y_max[k], y_step[k],
                                          time_start[index]   , time_stop[index]   )                                     
            else:              
                plot4par = vis.lplots(ax, name_1[k], y_label[k], date_ind , nst, 
                                          l_p, y_min[k] , y_max[k] , y_step[k],
                                          time_start[index], time_stop[index] ) 

                                         
                                           
        elif clm_name[k] in ('T_2M', 'T_S'):                                   # Plot data T2M or TS
            print (f'Plot - {clm_name[k]}, period: {time_int_1} to {time_int_2}')
                                   
            if ldaily == daily_time_step:
                plot4par = vis.plots4(ax, cclm_mean[clm_name[k]]    ,  cclm_v35_mean[clm_name[k]],
                                          cclm_v45_mean[clm_name[k]], cclm_v45e_mean[clm_name[k]],
                                          'CCLMref'   , 'CCLMv3.5'  , 'CCLMv4.5'    , 'CCLMv4.5e') 
                
                plot4par = vis.dplots(ax, name_1[k], y_label[k], name_3  , date_ind ,  
                                          nst, l_p ,  y_min[k], y_max[k], y_step[k],
                                          time_start[index]   , time_stop[index]   )
                                                                                       
            else:
                plot4par = vis.plots5(ax, cclm_mean[clm_name[k]]    , cclm_v35_mean[clm_name[k]]          , 
                                          cclm_v45_mean[clm_name[k]], cclm_v45e_mean[clm_name[k]]         , 
                                          t2m_hyras, 'CCLMref', 'CCLMv3.5', 'CCLMv4.5', 'CCLMv4.5e', 'OBS')
         
                plot4par = vis.lplots(ax, name_1[k], y_label[k], date_ind , nst, 
                                          l_p, y_min[k] , y_max[k] , y_step[k],
                                          time_start[index], time_stop[index] ) 
                
         

        elif clm_name[k] in ('AEVAP_S', 'ZVERBO'):                             # Plot data AEVAP_S or ZVERBO
            print (f'Plot - {clm_name[k]}, period: {time_int_1} to {time_int_2}') 
            # Define in-situ data type
            if clm_name[k] in 'AEVAP_S':
                gleam_v35a = et_v35a
                gleam_v35b = et_v35b 
            else:
                gleam_v35a = ep_v35a  
                gleam_v35b = ep_v35b
            # -----------------------------------------------------------------
                    
            if ldaily == daily_time_step:
                plot4par = vis.plots4(ax, cclm_mean[clm_name[k]]    ,  cclm_v35_mean[clm_name[k]],
                                          cclm_v45_mean[clm_name[k]], cclm_v45e_mean[clm_name[k]],
                                          'CCLMref'   , 'CCLMv3.5'  , 'CCLMv4.5'    , 'CCLMv4.5e') 
                
                plot4par = vis.dplots(ax, name_1[k], y_label[k], name_3  , date_ind ,  
                                          nst, l_p ,  y_min[k], y_max[k], y_step[k],
                                          time_start[index]   , time_stop[index]   )                       
            else:
                plot4par = vis.plots6(ax, cclm_mean[clm_name[k]]    , cclm_v35_mean[clm_name[k]]       , 
                                          cclm_v45_mean[clm_name[k]], cclm_v45e_mean[clm_name[k]]      , 
                                          gleam_v35a  , gleam_v35b  , 'CCLMref', 'CCLMv3.5', 'CCLMv4.5', 
                                          'CCLMv4.5e' , 'GLEAM 3.5a', 'GLEAM 3.5b'                     )
         
                plot4par = vis.lplots(ax, name_1[k], y_label[k], date_ind , nst, 
                                          l_p, y_min[k] , y_max[k] , y_step[k],
                                          time_start[index], time_stop[index] )                 
                                                                                                                                
        elif clm_name[k] in ('ALHFL_S', 'ASHFL_S'):                            # Plot data ALHFL_S or ASHFL_S
            print (f'Plot - {clm_name[k]}, period: {time_int_1} to {time_int_2}')            
            # Define in-situ data type
            if clm_name[k] in 'ALHFL_S':
                heat_flux = le_flux
            else:
                heat_flux = h_euro            
            # -----------------------------------------------------------------                 
            plot4par = vis.plots5(ax, cclm_mean[clm_name[k]]    ,  cclm_v35_mean[clm_name[k]],
                                      cclm_v45_mean[clm_name[k]], cclm_v45e_mean[clm_name[k]],
                                      heat_flux, 'CCLMref', 'CCLMv3.5', 'CCLMv4.5', 'CCLMv4.5e', 'OBS')             
            
            
            if ldaily == daily_time_step:              
                plot4par = vis.dplots(ax, name_1[k], y_label[k], name_3  , date_ind ,  
                                          nst, l_p ,  y_min[k], y_max[k], y_step[k],
                                          time_start[index]   , time_stop[index]   )                                                                                                
            else:
                plot4par = vis.lplots(ax, name_1[k], y_label[k], date_ind , nst, 
                                          l_p, y_min[k] , y_max[k] , y_step[k],
                                          time_start[index], time_stop[index] )                  
                                  
                          
        output_name = f'{clm_name[k]}_{time_int_1}_{time_int_2}.png'
            
        plt.savefig(path_exit + output_name, format = 'png', dpi = 300) 

        plt.close(fig)        
        plt.gcf().clear() 
"""
       