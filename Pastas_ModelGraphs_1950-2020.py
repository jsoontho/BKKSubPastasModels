###############################################################################

# Developing Pastas models for different wells in Bangkok, Thailand
# A well nest at a given location may have 1-4 wells
# Simulating groundwater levels
# Calibration period: typically from 1978-2020 (based on data 
# availability)
# Inputs: Basin-wide Pumping
# Outputs: Pastas models (.pas files), graphs

# Jenny Soonthornrangsan 2023
# TU Delft

###############################################################################

###############################################################################
# import statements
###############################################################################

# Importing packages and libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pastas as ps
import os
import sys
import warnings

# Ignoring Pastas warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Changing current directory to locaiton of python script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Getting current directory path
abs_path = os.path.dirname(__file__)

# Importing script for pre-processing Thai GW data
import main_functions as js_mfs

# Turning off warnings
pd.options.mode.chained_assignment = None  # default='warn'

#%%############################################################################
# Plotting settings
###############################################################################

plt.rc('font', size=10) #controls default text size
plt.rc('axes', titlesize=10) #fontsize of the title
plt.rc('axes', labelsize=6) #fontsize of the x and y labels
plt.rc('xtick', labelsize=6) #fontsize of the x tick labels
plt.rc('ytick', labelsize=6) #fontsize of the y tick labels
plt.rc('legend', fontsize=6) #fontsize of the legend

#%%############################################################################
# Pastas settings
###############################################################################

# If saving model, save_model = 1
save_model = 0

# If importing previous saved models, import_model = 1
import_model = 1

# If saving graphs, save_graph = 1
save_graph = 1

# Publication graphs
paper_graph = 1

# Additional figures
add_graph = 0

# Folder to save/import graph and model 
modelpath = os.path.join(abs_path, 'models')
figpath = os.path.join(abs_path, 'figures')
pumppath = os.path.join(abs_path, 'inputs/BasinPumping.xlsx')
pumpsheet = 'EstTotalPump_54-60_Int50'

# If running with pumping
# If basin wide pumping set pump_basin_flag to 1
pump_basin_flag = 1

# Pumping response function
pump_rfunc = ps.Gamma

# Solver
# Options: ps.LmfitSolve, ps.LeastSquares
solver = ps.LeastSquares

# Noise model
noise_TF = True; 

# Option to run all well nests, set flag to 1
list_wellnest_flag = 1;

# If running only one well nest, set flag to 0, WellNest Name
Wellnest_name = ["LCBKK003"]

## Getting a list of all the wells
# Input path
input_path = 'inputs'

# Total path
tot_path = os.path.join(abs_path, input_path)

# All well nest list
if list_wellnest_flag == 1:
    
    files = os.listdir(tot_path)
    files = [i.replace('.xlsx', '') for i in files \
             if i.startswith('LC') and "_" not in i]
    
else:
    files = Wellnest_name

###############################################################################
# Creating Pastas Model
###############################################################################

# For each well nest
for Wellnest_name in files:
    
    # Only open figures for each well nest
    plt.close("all")
    
    # Reading in groundwater data
    full_path = os.path.join(tot_path, Wellnest_name + '.xlsx')
    data = pd.read_excel(full_path, skiprows=3)
    
    # For all wells in well nest
    for wells in data.columns[-(len(data.columns)-2):]:
        
        # Name of well as a string
        well_name = wells
        
        #######################################################################
        # Creating Pastas Model
        #######################################################################
        # If not importing, then creating new models
        if import_model == 0:
            
            # Preprocessing head data
            # Inputing in raw data for well nest and well name
            # Output: head data relative to 0 for the entire well nest
            #         head data relative to 0 m for specific well
            all_head_data, gw_well_head = js_mfs.GW_Data_Process(data, 
                                                                 well_name)
            print('\nGW obs imported for ' + Wellnest_name + ' and ' + \
                  well_name + '\n')
             
            # CORRECTING GW HEAD DATA TO LAND SURFACE (COASTAL DEM 2.1)
            # Reading in land surface elevation for each well nest
            landsurf_path = os.path.join(tot_path,
                                         'LandSurfElev_GWWellLocs.xlsx')
            landsurf_data = pd.read_excel(landsurf_path, 
                                    sheet_name='2.1', 
                                    usecols="C:F",
                                    index_col=0) 
            
            # Correcting head relative to 0 m to be head relative to a 
            # global datum. 
            # Corrected by adding land surface elevation values to 
            # head values relative to 0 m
            gw_well_head.Head += (landsurf_data.RASTERVALU.loc[Wellnest_name])
            
            # Saving years and annual average heads
            gw_well_head["year"] = gw_well_head.index.year # Saving year
            gw_year = gw_well_head.groupby(gw_well_head["year"]).mean()
            gw_year["Date"] = pd.to_datetime(gw_year.index, format='%Y')
            gw_year.index = gw_year.Date # Setting index as date
            gw_year["year"] = gw_year.index.year # Saving year
            
            # Setting time min and time max to first and last obs year
            time_min = str(gw_year[gw_year.Head.notna()].year[0])
            time_max = str(gw_year[gw_year.Head.notna()].year[-1])
  
            # Gets rid of data not within min and max time
            gw_year = gw_year[(gw_year['year']>=int(time_min)) & \
                              (gw_year['year']<=int(time_max))]
            
            # If GW well does not have data within time period, skips this well
            if gw_year.Head.isnull().all():
                print("\nSkipping ", well_name, 
                      "because no head values in time period\n\n\n")
                continue
            
            # Initializing model
            model = ps.Model(gw_well_head.Head)
            print("\nModel initialized for GW\n")
            
            ## Initial steady state heads estimate for d constant
            # Steady state heads already realtive to same global datum
            # as land surface elevation
            SS_path = os.path.join(tot_path,
                                   'SS_Head_GWWellLocs.xlsx')
            SS_data = pd.read_excel(SS_path, 
                                    sheet_name='SS_Py', 
                                    index_col=0) 
            
            # Getting steady state heads according to aquifer
            if "BK" in well_name:
                
                initial_d = SS_data.loc[Wellnest_name, "BK"]
              
            elif "PD" in well_name:
                
                initial_d = SS_data.loc[Wellnest_name, "PD"]
            
            elif "NL" in well_name:
                
                initial_d = SS_data.loc[Wellnest_name, "NL"]
            
            elif "NB" in well_name:
                
                initial_d = SS_data.loc[Wellnest_name, "NB"]
                
            # Setting d parameter to SS heads and to vary +/- initial
            # estimates
            model.set_parameter(name="constant_d", 
                                initial=initial_d,
                                pmin=initial_d-10, 
                                pmax=initial_d+10,
                                vary=True)
                    
            ###################################################################
            # Adding pumping data
            ###################################################################
            if pump_basin_flag == 1:
               
                ## Daily interpolated and estimated pumping rates for the basin 
                # from simulated (Chula report)
                EstTotPump = pd.read_excel(pumppath, sheet_name=pumpsheet, 
                                           index_col=0, parse_dates=['Date'])
                
                # Creating stress model
                EstTotPump_ = ps.StressModel(EstTotPump.Pump, rfunc=pump_rfunc, 
                                             name="well", settings="well", 
                                             up=False)
                # Adding stress model
                model.add_stressmodel(EstTotPump_)
                print('\nPumping obs added basin-wide\n')
                
            ###################################################################
            # Solving/Saving Pastas Model
            ###################################################################
            # Try running model
            try:
                print("Running....")
                
                 # If noise model True
                if noise_TF == True:
                    
                    # First run is not with noise model
                    # Gets first parameter estimates
                    # Warm up is 30 years
                    model.solve(tmin=time_min, tmax=time_max, 
                                report="basic", noise=False, 
                                solver=solver, warmup=365*30)
                    
                    # Second run with noise model using initial 
                    # parameters as the calibrated parameters from 
                    # first run
                    model.solve(tmin=time_min, tmax=time_max, 
                                initial=False, report="basic", 
                                noise=noise_TF, solver=solver,
                                warmup=365*30)
                
            # If time series out of bounds
            except ValueError:
                print('Time series out of bounds.\nCannot run model')
                sys.exit()
            
            # If saving model
            if save_model == 1:
                model.to_file(modelpath + "\\" + Wellnest_name + '_' + \
                              well_name + '_GW_' + time_min + '_' + time_max +\
                              '_model.pas')
        
        #######################################################################
        # Importing Pastas Model
        #######################################################################
        # If importing Pastas model
        else:
            
            # Model files
            modelfiles = os.listdir(modelpath)
            
            # If file exists:
            try:
                
                # Load existing model
                wellmodel = [s for s in modelfiles \
                             if np.logical_and(Wellnest_name in s, well_name in s)][0]
                model = ps.io.load(modelpath + "\\" + wellmodel)
                   
                # Gets time min and max from file name
                time_min = wellmodel[wellmodel.find("_1")+1:wellmodel.find("_1")+5]
                time_max = wellmodel[wellmodel.find("_2")+1:wellmodel.find("_2")+5]
            
                # Saving optimal parameters before deleting stress
                optiparam = model.parameters["optimal"]
                stdparam = model.parameters["stderr"]
                
                # Deleting stress
                model.del_stressmodel("well")
                
                # Adding new pumping stress time series
                # If the same pumping stress time series, then 
                # optimal parameters are the same
                EstTotPump = pd.read_excel(pumppath, sheet_name=pumpsheet, 
                                           index_col=0, parse_dates=['Date'])
                EstTotPump_ = ps.StressModel(EstTotPump.Pump, rfunc=pump_rfunc, 
                                             name="well", settings="well", 
                                             up=False)
                
                # Adding new stress model
                model.add_stressmodel(EstTotPump_)
                
                # Setting the same optimal parameters 
                model.parameters["optimal"] = optiparam
                model.parameters["stderr"] = stdparam
                
            # If does not exist
            except FileNotFoundError:
                print("No model for " + Wellnest_name + "_" + well_name)
                continue
        
        ###############################################################################
        # Simulating Pastas Model
        # Pastas Plotting and Graphing
        ###############################################################################

            # set plotting time min and time max
            if "BK" in well_name:
                ptime_min = '1986'
            else:
                ptime_min = '1978'
            ptime_max = "2020"
            
            # If plotting additional graphs
            if add_graph == 1:
                
                # Plotting for calibrated time period
                ax = model.plot(tmin=ptime_min, tmax=ptime_max, 
                                figsize=(10, 6));
                
                # Setting yaxis limits
                # Different for wells in different aquifers
                if "BK" in well_name:
                    plt.ylim(-25, -5)
                elif "PD" in well_name:
                    # Well nests with big variation in head
                    if Wellnest_name == "LCBKK036" or \
                        Wellnest_name == "LCSPK007":
                        plt.ylim(-55, -10)
                    # Rest of wells
                    else:
                        plt.ylim(-30, -5)
                elif "NL" in well_name:
                    plt.ylim(-70, -10)
                elif "NB" in well_name:
                    plt.ylim(-60, -5)
            
                # If saving graphs
                if save_graph == 1:
                    
                    # First figure from plot
                    # Fig name
                    fig_name1 = Wellnest_name + '_' + well_name + '_GW_' + \
                                time_min + '_' + time_max + '_1.png'
                    # Fig path
                    full_figpath = os.path.join(figpath, fig_name1)
                    # Saving fig
                    plt.savefig(full_figpath, dpi=150, bbox_inches='tight', 
                                format='png')
                    
                    # Second figure
                    model.plots.results(tmin=ptime_min, tmax=ptime_max, 
                                        figsize=(10, 6));
                    # Fig name
                    fig_name2 = Wellnest_name + '_' + well_name + '_GW_' + time_min + '_' + time_max + '_2.png'
                    # Fig path
                    full_figpath = os.path.join(figpath, fig_name2)
                    # Saving fig
                    plt.savefig(full_figpath, dpi=150, bbox_inches='tight', 
                                format='png')
                
                    
                # If not saving graphs
                else:
                    #model.plots.results(tmin=time_min, tmax=time_max, figsize=(10, 6), ci=ci);
                    model.plots.results(tmin=time_min, tmax=time_max, figsize=(10, 6));

            # If replicating publication figures
            if paper_graph == 1:
                
                # Obs, simulation, residuals, stress time series
                # Observation time series
                o = model.observations(tmin=ptime_min, tmax=ptime_max)
                o_nu = model.oseries.series.drop(o.index)
                o_nu = o_nu[ptime_min: ptime_max]
                
                # Simulation time series
                sim = model.simulate(tmin=ptime_min, tmax=ptime_max, 
                                     warmup=365*30, return_warmup=False)
                
                # Residual time series
                res = model.residuals(tmin=ptime_min, tmax=ptime_max)
                
                # Getting stress contributions
                contribs = model.get_contributions(split=False, tmin=ptime_min,
                                         tmax=ptime_max,return_warmup=False)
                
                # Setting y limits
                ylims = [(min([sim.min(), o[ptime_min:ptime_max].min()]),
                          max([sim.max(), o[ptime_min:ptime_max].max()])),
                         (res.min(), res.max())]  # residuals are bigger than noise
                
                # Calculates height ratios for plots based on y limits
                def _get_height_ratios(ylims):
                    
                    height_ratios = []
                    for ylim in ylims:
                        hr = ylim[1] - ylim[0]
                        if np.isnan(hr):
                            hr = 0.0
                        height_ratios.append(hr)
                    return height_ratios
                
                # Gets the only contribution (pumping)
                contrib = contribs[0]
                hs = contrib.loc[ptime_min:ptime_max]
                ylims.append((hs.min(), hs.max()))
                
                # Calculate height ratio
                hrs = _get_height_ratios(ylims)
                
                # Creating figure
                fig = plt.figure(figsize=(3.3, 2.2), dpi=300)
                
                # Adding subfigures
                gs = fig.add_gridspec(ncols=1, nrows=3,
                               width_ratios=[.25], height_ratios=hrs)
        
                # Main frame
                ax1 = fig.add_subplot(gs[0])
                
                # First subplot
                # observation plot
                o.plot(ax=ax1, linestyle='-', marker='.', 
                       color='k', label='Observed Head', 
                       x_compat=True, markersize=3)
                if not o_nu.empty:
                    # plot parts of the oseries that are not used in grey
                    o_nu.plot(ax=ax1, linestyle='-', marker='.', color='0.5', 
                              label='',
                              x_compat=True, zorder=-1,
                              markersize=3)
        
                # add rsq to simulation
                rmse = model.stats.rmse(tmin=ptime_min, tmax=ptime_max)
                
                # Simulation plot
                sim.plot(ax=ax1, x_compat=True, 
                         label=f'Simulated (RMSE = {rmse:.2} m)', 
                         linewidth=1.5)
                
                # Plot 1 settings
                ax1.set_ylabel("Head (m)", labelpad=0)
                ax1.legend(loc=(0, 1), ncol=3, frameon=False, numpoints=3)
                ax1.set_ylim(ylims[0])
                ax1.tick_params(axis='both', which='major', pad=0)
                
                plt.annotate(Wellnest_name + " " + well_name, 
                             xy=(.5, 1.43),xycoords='axes fraction', 
                             fontsize=8, horizontalalignment='center', 
                             verticalalignment='top')
             
                # Second subplot
                # Residuals and noise
                ax2 = fig.add_subplot(gs[1], sharex=ax1)
                res.plot(ax=ax2, color='k', x_compat=True)
                
                # Adds noise model
                if model.settings["noise"] and model.noisemodel:
                    noise = model.noise(tmin=ptime_min, tmax=ptime_max)
                    noise.plot(ax=ax2, x_compat=True)
                    
                # Plottign setttigns
                ax2.axhline(0.0, color='k', linestyle='--', zorder=0)
                ax2.legend(loc=(0, 1), ncol=3, frameon=False)
                ax2.tick_params(axis='both', which='major', pad=0)
        
                # Add a row for each stressmodel
                rmin = 0  # tmin of the response
                rmax = 0  # tmax of the response
                axb = None
                i = 0
                
                # For each stress model
                for sm_name, sm in model.stressmodels.items():
                    
                    # Subplots for stress
                    ax = fig.add_subplot(gs[i + 2], sharex=ax1)
                    
                    # stress contribution plot
                    contribs[i].plot(ax=ax, x_compat=True)
                    
                    # Title of stress
                    title = [stress.name for stress in sm.stress]
                    if len(title) > 3:
                        title = title[:3] + ["..."]
                        
                    # Plotting settigns
                    ax.set_title(f"Stresses: {title}", loc="right",
                                 fontsize=plt.rcParams['legend.fontsize'],
                                 pad=3)
                    ax.tick_params(axis='both', which='major', pad=0)
                    ax.set_xlabel("Year", labelpad=0)
                    ax.set_ylabel("Head (m)", labelpad=0)
                    ax.legend(loc=(0, 1), ncol=3, frameon=False)
                    ax.set_ylim(ylims[i + 2])
                    i = i + 1
        
                    # plot the step response
                    response = model._get_response(block_or_step="step",
                                                     name="well", add_0=True)
                    
                    rmax = max(rmax, response.index.max())
                    
                    # Inset graph settings
                    left, bottom, width, height = [0.79, 0.19, 0.15, 0.1]
                    axb = fig.add_axes([left, bottom, width, height])
                    response.plot(ax=axb)
                    title = 'Step response'
                    axb.tick_params(axis='both', which='major', pad=0)
                    axb.set_xlabel("", fontsize=5, labelpad=0)
                    axb.set_title(title, fontsize=5, pad=0)
                    axb.tick_params(labelsize=5)
                    axb.set_xlim(rmin, rmax)
        
                # xlim sets minorticks back after plots:
                ax1.minorticks_off()
        
                ax1.set_xlim(time_min, time_max)
        
                # sometimes, ticks suddenly appear on top plot, turn off just in case
                plt.setp(ax1.get_xticklabels(), visible=False)
        
                # Grids
                for ax in fig.axes:
                    ax.grid(True)
                # No grids for inset graph
                axb.grid(False)
                
                if isinstance(fig, plt.Figure):
                    fig.tight_layout(pad=0)  # Before making the table
                
                plt.subplots_adjust(right = 0.95)
                
                # Fig name
                fig_name3 = Wellnest_name + '_' + well_name + '_GW_' + \
                            ptime_min + '_' + ptime_max + '_PAPER.png'
                # Fig path
                full_figpath = os.path.join(figpath, fig_name3)
                # Save fig
                plt.savefig(full_figpath, dpi=300, bbox_inches='tight', 
                            format='png')
            