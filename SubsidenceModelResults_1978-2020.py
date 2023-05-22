# ##############################################################################
"""Calculate subsidence in BKK at wellnests with 8 aquifers but simulates top four.

BK, PD, NL, NB
all are confined and overlain by clay layer
Implicit method according to USGS SUB package Hoffman report pg. 14

Output:

1. Bar graphs of annual subsidence (cm) for each well nest during 1978-2020
(Shown in the main text and supplemental information)
2. Line graphs of annual subsidence (cm) for sensitivity analyses of each parameter
(Sskv, Sske, K, thickness) for one well nest (long run time so only calculating for
one well nest at a time) (Shown in supplemental information)
3. Line graphs of cumulative subsidence (cm) into the future depending on the
pumping scenario for each well nest during 1978-2060 (Shown in the main text and
supplemental information)

Jenny Soonthornrangsan 2023
TU Delft
"""
# ##############################################################################

###############################################################################
# import statements
###############################################################################

import os
import pandas as pd
import numpy as np

# Bangkok Subsidence Model Package
import bkk_sub_gw

# %%###########################################################################
# Runs the functions to calculate subsidence at point locations in BKK
# Main paper graph
##############################################################################

# For well nest BKK013 (in paper) = LCBKK013
wellnestlist = ["LCBKK013"]
tmin = "1978"
tmax = "2020"

# Reading in thickness and storage data
path = os.path.join(os.path.abspath("inputs"), "SUBParameters.xlsx")
Thick_data = pd.read_excel(path, sheet_name="Thickness",
                           index_col=0)  # Thickness
Sskv_data = pd.read_excel(path,
                          sheet_name="Sskv",
                          index_col=0)  # Sskv
Sske_data = pd.read_excel(path,
                          sheet_name="Sske",
                          index_col=0)  # Ssk
K_data = pd.read_excel(path,
                       sheet_name="K",
                       index_col=0)  # K

# Mode can be "raw" as in raw groundwater data vs "Pastas" for importing Pastas
# simulated groundwater in the aquifers
mode = "Pastas"

# If mode is Pastas, need model path
if mode == "Pastas":

    mpath = os.path.abspath("models")

# Pumping flag, for PASTAS, if changing pumping scenario
pumpflag = 1
# If changing pumping scenario, need pumping sheet/path
if pumpflag == 1:

    ppath = os.path.join(os.path.abspath("inputs"), "BasinPumping.xlsx")
    psheet = "EstTotalPump_54-60_Int50"

# Convergence criteria
CC = 1 * 10**-5

# Number of nodes in clay
node_num = 10

# Using available heads as proxy for missing
proxyflag = 1

# Calculates subsidence
all_results, sub_total, subv_total = bkk_sub_gw.\
    bkk_sub.bkk_subsidence(wellnestlist,
                           mode, tmin,
                           tmax,
                           Thick_data,
                           K_data,
                           Sskv_data,
                           Sske_data,
                           CC=CC,
                           Nz=node_num,
                           ic_run=True,
                           proxyflag=proxyflag,
                           pumpflag=pumpflag,
                           pump_path=ppath,
                           pump_sheet=psheet,
                           model_path=mpath)

# Post process data
sub_total, subv_total, ann_sub, \
    avgsub = bkk_sub_gw.bkk_sub.bkk_postproc(wellnestlist,
                                             sub_total,
                                             subv_total,
                                             all_results)

# Plotting
# path to save figures
path = os.path.abspath("figures")

###############################################################################
# Plots Results: Bar graph for main paper for BKK013
##############################################################################

bkk_sub_gw.bkk_plotting.sub_bar(path, wellnestlist, all_results,
                                sub_total, subv_total, ann_sub,
                                tmin=tmin, tmax=tmax, save=1,
                                benchflag=1)

# %%###########################################################################
# Runs the functions to calculate subsidence at point locations in BKK
# Appendix graphs
##############################################################################

# For each well nest
wellnestlist = ["LCBKK003",
                "LCBKK005",
                "LCBKK006",
                "LCBKK007",
                "LCBKK009",
                "LCBKK011",
                "LCBKK012",
                "LCBKK013",
                "LCBKK014",
                "LCBKK015",
                "LCBKK016",
                "LCBKK018",
                "LCBKK020",
                "LCBKK021",
                "LCBKK026",
                "LCBKK027",
                "LCBKK036",
                "LCBKK038",
                "LCBKK041",
                "LCNBI003",
                "LCNBI007",
                "LCSPK007",
                "LCSPK009"]

# Calculates subsidence
all_results, sub_total, subv_total = bkk_sub_gw.\
    bkk_sub.bkk_subsidence(wellnestlist,
                           mode, tmin,
                           tmax,
                           Thick_data,
                           K_data,
                           Sskv_data,
                           Sske_data,
                           CC=CC,
                           Nz=node_num,
                           ic_run=True,
                           proxyflag=proxyflag,
                           pumpflag=pumpflag,
                           pump_path=ppath,
                           pump_sheet=psheet,
                           model_path=mpath)

# Post process data
sub_total, subv_total, ann_sub, \
    avgsub = bkk_sub_gw.bkk_sub.bkk_postproc(wellnestlist,
                                             sub_total,
                                             subv_total,
                                             all_results)

# Average perc of each clay layer to total for all well nest
BKClayavg = np.average([i[2] for i in avgsub[0::4]])*100
PDClayavg = np.average([i[2] for i in avgsub[1::4]])*100
NLClayavg = np.average([i[2] for i in avgsub[2::4]])*100
NBClayavg = np.average([i[2] for i in avgsub[3::4]])*100

# Plotting
# path to save figures
path = os.path.abspath("figures")

# %%###########################################################################
# Plots Results: Bar graph for appendix
##############################################################################

bkk_sub_gw.bkk_plotting.sub_bar(path, wellnestlist, all_results,
                                sub_total, subv_total, ann_sub,
                                tmin=tmin, tmax=tmax, save=1,
                                benchflag=1)

# %%###########################################################################
# Plots Results: Subsidence RMSE map for main paper
##############################################################################

# Spatial map plotting
bkk_sub_gw.bkk_plotting.sub_rmse_map(path, wellnestlist, all_results,
                                     sub_total, subv_total,
                                     ann_sub, tmin=tmin, tmax=tmax, save=1)

# %%###########################################################################
# Plots Results: Forecasts of cumulative subsidence (cm) for pumping scenarios
##############################################################################

# For each well nest
wellnestlist = ["LCBKK003",
                "LCBKK005",
                "LCBKK006",
                "LCBKK007",
                "LCBKK009",
                "LCBKK011",
                "LCBKK012",
                "LCBKK013",
                "LCBKK014",
                "LCBKK015",
                "LCBKK016",
                "LCBKK018",
                "LCBKK020",
                "LCBKK021",
                "LCBKK026",
                "LCBKK027",
                "LCBKK036",
                "LCBKK038",
                "LCBKK041",
                "LCNBI003",
                "LCNBI007",
                "LCSPK007",
                "LCSPK009"]
tmin = "1978"
tmax = "2060"

# Pumping flag, for PASTAS, if changing pumping scenario
pumpflag = 1
# If changing pumping scenario, need pumping sheet/path
if pumpflag == 1:

    ppath = os.path.join(os.path.abspath("inputs"), "BasinPumping.xlsx")

    # Pumping sheets
    pumpsheets = ["EstTotalPump_54-60_Int50",
                  "EstTotalPump_54-60_IntF25",
                  "EstTotalPump_54-60_IntF100",
                  "EstTotalPump_54-60_IntF50_25",
                  "EstTotalPump_54-60_IntF0"]

# Convergence criteria
CC = 1 * 10**-5

# Number of nodes in clay
node_num = 10

# Using available heads as proxy for missing
proxyflag = 1

# All ann subs
all_ann_subs = []

# For each pumping scenario
for pumpsheet in pumpsheets:

    # Calculates subsidence
    all_results, sub_total, subv_total = bkk_sub_gw.\
        bkk_sub.bkk_subsidence(wellnestlist,
                               mode, tmin,
                               tmax,
                               Thick_data,
                               K_data,
                               Sskv_data,
                               Sske_data,
                               CC=CC,
                               Nz=node_num,
                               ic_run=True,
                               proxyflag=proxyflag,
                               pumpflag=pumpflag,
                               pump_path=ppath,
                               pump_sheet=pumpsheet,
                               model_path=mpath)

    # Post process data
    sub_total, subv_total, ann_sub, \
        _ = bkk_sub_gw.bkk_sub.bkk_postproc(wellnestlist,
                                            sub_total,
                                            subv_total,
                                            all_results)

    all_ann_subs.append(ann_sub)

# Plotting
# path to save figures
path = os.path.abspath("figures")

# %%###########################################################################
# Plots Results: Line graphs of cumulative sub forecast for whole time period
# For appendix
##############################################################################

bkk_sub_gw.bkk_plotting.sub_forecast(path, wellnestlist, all_ann_subs,
                                     save=1)

# %%###########################################################################
# Plots Results: Maps of cumulative sub forecast from new tmin and tmax
# For main paper
##############################################################################

tmin = "2020"
tmax = "2060"
bkk_sub_gw.bkk_plotting.sub_forecast_map(path, wellnestlist,
                                         all_ann_subs, tmin, tmax,
                                         save=1)

# %%###########################################################################
# Plots Results: Sensitivity Analysis, shown in appendix
##############################################################################

# Pumping flag, for PASTAS, if changing pumping scenario
pumpflag = 1
# If changing pumping scenario, need pumping sheet/path
if pumpflag == 1:

    ppath = os.path.join(os.path.abspath("inputs"), "BasinPumping.xlsx")
    psheet = "EstTotalPump_54-60_Int50"

# Convergence criteria
CC = 1 * 10**-5

# Number of nodes in clay
node_num = 10

# Using available heads as proxy for missing
proxyflag = 1

# Recommended looking at results from sensitivity analysis for only one well nest
# Well nest to run sensitivity analysis
wellnest_sens = ["LCBKK013"]

# Sensitivity analysis
sens_modes = ["Sske_clay", "thick", "Sskv", "K", "Sske_sand"]

# For each sensitivity parameter set
for sens_mode in sens_modes:

    tmin = "1978"
    tmax = "2060"

    # Increasing by 10%
    coeff = .5
    num = 11  # Num of increases in percentage

    # Preallocation
    # All results from every sensitivity
    sens_results = []
    sens_sub = []
    sens_subv = []
    sens_ann = []

    # For each parameter increase
    for i in range(num):

        # Reading in thickness and storage data
        path = os.path.join(os.path.abspath("inputs"), "SUBParameters.xlsx")
        Thick_data = pd.read_excel(path, sheet_name="Thickness",
                                   index_col=0)  # Thickness
        Sskv_data = pd.read_excel(path,
                                  sheet_name="Sskv",
                                  index_col=0)  # Sskv
        Sske_data = pd.read_excel(path,
                                  sheet_name="Sske",
                                  index_col=0)  # Ssk
        K_data = pd.read_excel(path,
                               sheet_name="K",
                               index_col=0)  # K

        # Sensitivity analyses depending on parameter
        # Inelastic specific storage
        if sens_mode == "Sskv":

            Sskv_data = Sskv_data.iloc[:, :9] * coeff

        # Elastic specific storage for clay
        elif sens_mode == "Sske_clay":

            Sske_data.iloc[:, 0:9:2] = Sske_data.iloc[:, 0:9:2] * coeff

        # Elastic specific storage for sand
        elif sens_mode == "Sske_sand":

            # If not the last sens
            if i != (num - 1):

                Sske_data.iloc[:, 1:10:2] = Sske_data.iloc[:, 1:10:2] * coeff

            # If last sens, setting sand elastic storage to clay which is typically
            # one order of magnitude higher
            else:

                Sske_data.iloc[:, 1:10:2] = Sske_data.iloc[:, 0:9:2]

        # Vertical hydraulic conductivity
        elif sens_mode == "K":

            K_data = K_data.iloc[:, :9] * coeff

        # Thickness
        elif sens_mode == "thick":

            Thick_data = Thick_data.iloc[:, :9] * coeff

        # Running subsidence model for every analysis value
        all_, sub_, subv_ = bkk_sub_gw.\
            bkk_sub.bkk_subsidence(wellnest_sens,
                                   mode, tmin,
                                   tmax,
                                   Thick_data,
                                   K_data,
                                   Sskv_data,
                                   Sske_data,
                                   CC=CC,
                                   Nz=node_num,
                                   ic_run=True,
                                   proxyflag=proxyflag,
                                   pumpflag=pumpflag,
                                   pump_path=ppath,
                                   pump_sheet=psheet,
                                   model_path=mpath)

        sub_, subv_, ann_, _ = bkk_sub_gw.bkk_sub.bkk_postproc(wellnest_sens,
                                                               sub_,
                                                               subv_,
                                                               all_)

        # Saving results
        sens_results.append(all_)
        sens_sub.append(sub_)
        sens_subv.append(subv_)
        sens_ann.append(ann_)

        # Shifting parameter value
        coeff += .1

    # Plotting
    # path to save figures
    path = os.path.abspath("figures")

    # Plots results
    # New tmin for subsidence change
    tmin = "2020"
    tmax = "2060"
    bkk_sub_gw.bkk_plotting.sub_sens_line(path, wellnest_sens, sens_results,
                                          sens_sub, sens_subv, sens_ann,
                                          tmin=tmin, tmax=tmax, mode=sens_mode,
                                          num=num, save=1)
