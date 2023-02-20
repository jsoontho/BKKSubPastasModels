"""

Script for Thailand GW data pre-processing and post-plotting

Thai GW data from http://tgms.dgr.go.th/
Jenny Soonthornrangsan

"""
###############################################################################
# import statements
###############################################################################

# import statements
import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge

# Closing all figures
plt.close('all')

###############################################################################
###############################################################################
###############################################################################

# String formatting
# Function to preprocess groundwater data from http://tgms.dgr.go.th/#/home
# Assuming excel format
def GW_Data_Process(GW_Data, well_name=None):
    # # Inputs:
    # GW data assuming as dataframe and first five lines irrevelevant with - to be replaced with nan
    # Thai years need to be converted to english years
    # Well name - if None, returning all wells with same dates
    # # Outputs:
    # Data frame with dates and groundwater data (HEAD REL to 0 NOT DEPTH TO WATER)
    # Subset of data for only the well from well_name - If well name not None
    
    # Ignoring first five lines
    # Replacing no date (-) with nans
    data = GW_Data.iloc[0:len(GW_Data)-5, :];
    data = data.replace('-', np.nan)
    
    # Creating data frame
    well_data = data.iloc[:, 1:]
    df_data = well_data.rename(columns={'วันที่': 'Date'})
    df_data.Date = df_data['Date'].astype(str)
    
    # Reformating date from thai years to english years
    len_head = len(df_data.iloc[:, 1])
    date_list = []
    df_data['Year'] = np.nan
    df_data['Month'] = np.nan
    df_data['Day'] = np.nan
    
    # For each data point
    for i in range(len_head):
        
        # Current date
        date = df_data.Date.loc[i]

        # If leap day 
        if date[0:2] == '29' and date[3:5] == '02':
            
            # Thai years - 543 = english years
            df_data.Year.loc[i] = int(date[6:10]) - 543;
            df_data.Month.loc[i] = int(date[3:5])
            df_data.Day.loc[i] = int(date[0:2])
            
            # Converting to date time format
            date_list.append(dt.datetime.strptime(str(df_data.Day.loc[i]).replace(".0", "") + "/" \
                                                  + str(df_data.Month.loc[i]).replace(".0", "") + \
                                                  "/" + str(df_data.Year.loc[i]).replace(".0", ""), \
                                                  "%d/%m/%Y").date());
        
        # If not a leap date
        else:
            
            # Converting to date time format
            date_list.append(dt.datetime.strptime(date, "%d/%m/%Y %M:%S").date())
            df_data.Year.loc[i] = (date_list[i].year) - 543
            df_data.Month.loc[i] = (date_list[i].month)
            df_data.Day.loc[i] = (date_list[i].day)
        
    # Saving new english date
    df_data['EngDate'] = pd.to_datetime(df_data[['Year', 'Month', 'Day']]);
    
    # If individual well name given
    if well_name != None:
        
        # Subset of dataframe to get ready for Pastas
        # DTW data converted to approximate head using negative sign
        head_subsetdata = {'Date': df_data.EngDate,
                'Head': - df_data[well_name]}
        Head_Data = pd.DataFrame(head_subsetdata, columns = ['Date', 'Head'])
        Head_Data.index = pd.to_datetime(Head_Data.Date)
    
    # If individual head date not given
    else:
        Head_Data = None
        
    # Cleaning all data up
    # Returning head data not depth to water!
    all_data = pd.concat([df_data['EngDate'], -well_data.iloc[:, 1:]], axis=1,\
                        keys=['Date'] + well_data.columns.values[1:])
    if (np.size(-well_data.iloc[:, 1:], axis=1)) > 1:
        all_data = all_data.droplevel(level=0, axis=1)
        
    return all_data, Head_Data

###############################################################################
###############################################################################
###############################################################################

# Checks for outliers
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if np.size(points) == 1:
        points = points[:,None]
    median = np.median(points)
    diff = (points - median)**2
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

###############################################################################
###############################################################################
###############################################################################

# Drawing basemap for BKK
def draw_basemap(map, xs, ys, cs, fig=None, ax=None,
                 datalim=None, mode=None, save=0, aq=None, 
                 perc=None, time_min=None, time_max=None, figpath=None, crit=None):
    
    # Mode can be RMSE_full (Pastas), step_full (Pastas t90)
    # sub_RMSE (subsidence RMSE), sub_forecast (subsidence forecast)
    # Map contains the basemap

    ## Plotting map
    # Adding shaded relief background
    map.shadedrelief()

    # Drawing coastline
    map.drawcoastlines(zorder = 2, linewidth = 1)
    
    # Adding Thailand province boundaries
    map.readshapefile(r'C:\Users\jtsoonthornran\Downloads\BKK_Shapefiles\BKK_Provinces\tha_admbnda_adm1_rtsd_20190221', \
                      name='provinces', drawbounds=True, zorder = 1, linewidth = .5)
    
    # Drawing rivers
    map.drawrivers(color='teal', linewidth=1)
    
    # draw parallels and meridians
    map.drawparallels(np.arange(12.5,14,.5),labels=[1,0,0,0],
                      fontsize=6)
    map.drawmeridians(np.arange(99.5,101.5,.25),labels=[0,0,0,1],
                      fontsize=6)

    # Drawing Pastas/subsidence datapoints
    x, y = map(xs, ys)

    # RMSE mode
    if mode == 'RMSE_full':
            
        # Angle of wedges
        theta1 = 90; theta2 = 180
        r = .018 # radius
        
        # For each well
        for item in cs.items():
            
            # Patches
            patches = []
            
            # For each location
            for j in range(len(item[1].x)):
                
                # Creating wedge
                wedge = Wedge((item[1].x[j], item[1].y[j]), 
                              r, theta1, theta2)
                patches.append(wedge) 
            
            # Adding collection
            p = PatchCollection(patches, zorder=10,
                                edgecolor="k",
                                linewidth=.5)
            p.set_array(item[1].cs) # Colorbar
            ax.add_collection(p)
            
            # Updating theta
            theta1 -= 90; theta2 -= 90
        
        # Colorbar
        
        cb = fig.colorbar(p, ax=ax); 
        cb.ax.tick_params(labelsize=6) 
        cb.set_label("RMSE %", fontsize=7)
        plt.set_cmap("coolwarm")
        cb.mappable.set_clim(vmin=datalim[0], 
                          vmax=datalim[1])
        cb.solids.set_rasterized(False) 
    
        # Legend objects
        class WedgeObject(object):
            pass
        
        class WedgeObjectHandler(object):
            
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                x0, y0 = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                hw = x0+.45*width; hh = y0+.5*height
                r2 = 5
                colors = ['#42BCFF',"#FF8300",'#D30808','#009914']
                lgd_patches = [Wedge((hw, hh), r2, 90, 180,  color=colors[0], label='BK'),
                                Wedge((hw, hh), r2, 0, 90, color=colors[1], label='PD'),
                                Wedge((hw, hh), r2, 180, 270, color=colors[2], label='NB'),
                                Wedge((hw, hh), r2, 270, 360, color=colors[3], label='NL')]
            
                lgd_elements = PatchCollection(lgd_patches,
                                        match_original=True,
                                        edgecolor="k",
                                        linewidth=.5)
        
                handlebox.add_artist(lgd_elements)
                plt.legend()
                return lgd_elements
        

        plt.legend([WedgeObject()], ['BK PD\nNB NL'],
           handler_map={WedgeObject: WedgeObjectHandler()},
           fontsize=5)
        
        
        # Title
        avgRMSE = pd.concat([cs["NB"].cs, cs["NL"].cs, cs["PD"].cs, 
                             cs["BK"].cs], ignore_index=True)
        plt.title("Average " + str("%.2f" % np.average(avgRMSE)) + \
                  "% of RMSE \nRelative to Observation Range", 
                  {'fontname':'Arial'}, fontsize=8)
            
    elif mode == 'step_full':
        
        # Angle of wedges
        theta1 = 90; theta2 = 180
        r = .018 # radius
        
        # For each well
        for item in cs.items():
            
            # Patches
            patches = []
            
            # For each location
            for j in range(len(item[1].x)):
                
                # Creating wedge
                wedge = Wedge((item[1].x[j], item[1].y[j]), 
                              r, theta1, theta2)
                patches.append(wedge) 
            
            # Adding collection
            p = PatchCollection(patches, zorder=10,
                                edgecolor="k",
                                linewidth=.5)
            p.set_array(item[1].cs) # Colorbar
            ax.add_collection(p)
            
            # Updating theta
            theta1 -= 90; theta2 -= 90
        
        # Colorbar
        
        cb = fig.colorbar(p, ax=ax); 
        cb.ax.tick_params(labelsize=6) 
        cb.set_label("Years", fontsize=7)
        cb.mappable.set_clim(vmin=datalim[0], 
                          vmax=datalim[1])
        plt.set_cmap("plasma")
        cb.solids.set_rasterized(False) 
        class Wedge_obj(object):
            pass
        
        class WedgeHandler(object):
            
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                x0, y0 = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                hw = x0+.45*width; hh = y0+.5*height
                r2 = 5
                colors = ['#42BCFF',"#FF8300",'#D30808','#009914']
                lgd_patches = [Wedge((hw, hh), r2, 90, 180,  color=colors[0], label='BK'),
                                Wedge((hw, hh), r2, 0, 90, color=colors[1], label='PD'),
                                Wedge((hw, hh), r2, 180, 270, color=colors[2], label='NB'),
                                Wedge((hw, hh), r2, 270, 360, color=colors[3], label='NL')]
            
                lgd_elements = PatchCollection(lgd_patches,
                                        match_original=True,
                                        edgecolor="k",
                                        linewidth=.5)
        
                handlebox.add_artist(lgd_elements)
                
                return lgd_elements
            
        plt.legend([Wedge_obj()], ['BK PD\nNB NL'],
           handler_map={Wedge_obj: WedgeHandler()},
           fontsize=5)
        
        
        # Title
        plt.title("$t_{90}$ for Step Response", {'fontname':'Arial'},
                  fontsize=8)
        
    elif mode == 'Sub_Forecast':
        
        scatter = map.scatter(x, y, s=300, 
                              c=np.multiply(cs, -1), zorder=3,
                              marker='o', edgecolor='k',
                              cmap='viridis_r')   
        plt.clim(datalim)
        cb = plt.colorbar()
        cb.set_label("Cumulative Subsidence (cm)")
        plt.title("Cumulative Subsidence 1978-2020", {'fontname':'Arial'})
        cb.solids.set_rasterized(False) 
        plt.show()
        
    elif mode == 'Sub_RMSE':
        
        scatter = map.scatter(x, y, s=50, 
                              c=np.multiply(cs, 1), zorder=3,
                              marker='o', edgecolor='k',
                              cmap='RdYlBu_r', linewidth=.75)   
        plt.clim(datalim)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=6) 
        cb.set_label("RMSE (cm)", fontsize=7)
        plt.title("RMSE between \nSimulated and Observed Subsidence", 
                  {'fontname':'Arial'}, fontsize=8)
        cb.solids.set_rasterized(False) 
        plt.show()
         
         
    # Annotating water bodies
    plt.annotate('Gulf of Thailand', xy=(.6, .05), 
                 xycoords='axes fraction', fontsize=5)
    plt.annotate('Chao Phraya River', xy=(.35, .95), 
                 xycoords='axes fraction', fontsize=5)
    # plt.annotate('Tha Chin River', xy=(.02, .95), xycoords='axes fraction')

    plt.show()

    # Saaving graphs
    if save == 1:
        fig_name1 = aq + '_' + mode + '_' + time_min + '_' + time_max + '_maps.png'
        full_figpath = os.path.join(figpath, fig_name1)
        plt.savefig(full_figpath, dpi=500, bbox_inches='tight')
        
###############################################################################
###############################################################################
###############################################################################
