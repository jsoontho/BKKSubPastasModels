# ##############################################################################
"""Calculate subsidence in BKK at wellnests with 8 aquifers but simulates top four.

BK, PD, NL, NB
all are confined and overlain by clay layer
Implicit method according to USGS SUB package Hoffman report pg. 14

Output:

1. Bar graph of annual subsidence (cm) for well nest BKK013 for 1979-2020
(Figure 10 of main text)
2. Bar graphs of annual subsidence (cm) for each well nest during 1978-2020
(Shown in supplemental information)
3. RMSE map of simulated annual subsidence vs observed from benchmark leveling
stations
4. Line graphs of cumulative subsidence (cm) into the future depending on the
pumping scenario for each well nest during 1978-2060 (Shown in the main text and
supplemental information)
5. Map of cumulative subsidence for 2020-2060 for each well nest for each
pumping scenario
6. Line graphs of annual subsidence (cm) for sensitivity analyses of each parameter
(Sskv, Sske, K, thickness) for one well nest (long run time so only calculating for
one well nest at a time) (Shown in supplemental information)


Article Title: Hybrid data-driven, physics-based modeling of ground-
water and subsidence with application to Bangkok, Thailand

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
import pickle
import datetime as dt
import warnings

# Bangkok Subsidence Model Package
import bkk_sub_gw

# Hampel filter
from hampel import hampel

# Ignoring Pastas warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# %%###########################################################################
# Runs the functions to calculate subsidence at point locations in BKK
# Main paper graph
##############################################################################

# Creating (0) or importing (1)
importing = 1

# If saving model
saving = 0

# For well nest BKK013 (in paper) = LCBKK013
wellnestlist = ["LCBKK013"]

# If creating results for first time
if importing == 0:

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

    # Dictionary to store everything
    model_sub = {"wellnestlist": wellnestlist,
                 "all_results": all_results,
                 "sub_total": sub_total,
                 "subv_total": subv_total,
                 "ann_sub": ann_sub,
                 "avgsub": avgsub,
                 "tmin": tmin,
                 "tmax": tmax,
                 "Thick_data": Thick_data,
                 "Sske_data": Sske_data,
                 "Sskv_data": Sskv_data,
                 "K_data": K_data,
                 "pumping_scenario": psheet,
                 "CC": CC,
                 "clay_nodes": node_num,
                 "proxyflag": proxyflag,
                 "mode": mode}

    # If saving model
    if saving == 1:

        # Path to save models
        path = os.path.abspath("models")

        # Saving dict for this model
        afile = open(path + "\\LCBKK013_sub.pkl", "wb")
        pickle.dump(model_sub, afile)
        afile.close()

# if importing subsidence model results
else:

    # Path to import models
    path = os.path.abspath("models")

    # Reload object from file
    file2 = open(path + "\\" + wellnestlist[0] + "_sub.pkl", "rb")
    model_sub = pickle.load(file2)
    file2.close()

# Plotting
# path to save figures
path = os.path.abspath("figures")

###############################################################################
# Plots Results: Bar graph for main paper for BKK013
##############################################################################

bkk_sub_gw.bkk_plotting.sub_bar(path, model_sub["wellnestlist"],
                                model_sub["all_results"],
                                model_sub["sub_total"],
                                model_sub["subv_total"],
                                model_sub["ann_sub"],
                                tmin=model_sub["tmin"],
                                tmax=model_sub["tmax"], save=1,
                                benchflag=1)

# %%###########################################################################
# Runs the functions to calculate subsidence at point locations in BKK
# Appendix graphs
##############################################################################

# Creating (0) or importing (1)
importing = 1

# If saving model
saving = 0

# If creating results for first time
if importing == 0:

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

    # Dictionary to store everything
    model_sub = {"wellnestlist": wellnestlist,
                 "all_results": all_results,
                 "sub_total": sub_total,
                 "subv_total": subv_total,
                 "ann_sub": ann_sub,
                 "avgsub": avgsub,
                 "tmin": tmin,
                 "tmax": tmax,
                 "Thick_data": Thick_data,
                 "Sske_data": Sske_data,
                 "Sskv_data": Sskv_data,
                 "K_data": K_data,
                 "pumping_scenario": psheet,
                 "CC": CC,
                 "clay_nodes": node_num,
                 "proxyflag": proxyflag,
                 "mode": mode}

    # If saving
    if saving == 1:

        # Path to save models
        path = os.path.abspath("models")

        # Saving dict for this model
        afile = open(path + "\\Allnests_sub.pkl", "wb")
        pickle.dump(model_sub, afile)
        afile.close()

# if importing subsidence model results
else:

    # Path to import models
    path = os.path.abspath("models")

    # Reload object from file
    file2 = open(path + "\\Allnests_sub.pkl", "rb")
    model_sub = pickle.load(file2)
    file2.close()

# Average perc of each clay layer to total for all well nest
BKClayavg = np.average([i[2] for i in model_sub["avgsub"][0::4]])*100
PDClayavg = np.average([i[2] for i in model_sub["avgsub"][1::4]])*100
NLClayavg = np.average([i[2] for i in model_sub["avgsub"][2::4]])*100
NBClayavg = np.average([i[2] for i in model_sub["avgsub"][3::4]])*100

list_ = ["LCBKK003", "LCBKK006", "LCBKK011", "LCBKK036", "LCBKK038"]
# Average perc of each clay layer to total for well nests with BK
BKClayavg_list1 = np.average([i[2] for i in model_sub["avgsub"][0::4]
                              if i[0] in list_])*100
PDClayavg_list1 = np.average([i[2] for i in model_sub["avgsub"][1::4]
                              if i[0] in list_])*100
NLClayavg_list1 = np.average([i[2] for i in model_sub["avgsub"][2::4]
                              if i[0] in list_])*100
NBClayavg_list1 = np.average([i[2] for i in model_sub["avgsub"][3::4]
                              if i[0] in list_])*100

# Average perc of each clay layer to total for well nests without BK
BKClayavg_list0 = np.average([i[2] for i in model_sub["avgsub"][0::4]
                              if i[0] not in list_])*100
PDClayavg_list0 = np.average([i[2] for i in model_sub["avgsub"][1::4]
                              if i[0] not in list_])*100
NLClayavg_list0 = np.average([i[2] for i in model_sub["avgsub"][2::4]
                              if i[0] not in list_])*100
NBClayavg_list0 = np.average([i[2] for i in model_sub["avgsub"][3::4]
                              if i[0] not in list_])*100

# Plotting
# path to save figures
path = os.path.abspath("figures")

# %%###########################################################################
# Plots Results: Bar graph for appendix
##############################################################################

bkk_sub_gw.bkk_plotting.sub_bar(path, model_sub["wellnestlist"],
                                model_sub["all_results"],
                                model_sub["sub_total"],
                                model_sub["subv_total"],
                                model_sub["ann_sub"],
                                tmin=model_sub["tmin"],
                                tmax=model_sub["tmax"], save=1,
                                benchflag=1)

# %%###########################################################################
# Plots Results: Subsidence RMSE map for main paper
##############################################################################

# Spatial map plotting
bkk_sub_gw.bkk_plotting.sub_rmse_map(path, model_sub["wellnestlist"],
                                     model_sub["all_results"],
                                     model_sub["sub_total"],
                                     model_sub["subv_total"],
                                     model_sub["ann_sub"],
                                     tmin=model_sub["tmin"],
                                     tmax=model_sub["tmax"], save=1)

# %%###########################################################################
# Plots Results: Forecasts of cumulative subsidence (cm) for pumping scenarios
##############################################################################

# Creating (0) or importing (1)
importing = 1

# If saving
saving = 0

# All ann subs
all_ann_subs = []

# If creating results for first time
if importing == 0:

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

        scenarios = ["500", "250", "100", "500_250", "0"]

    # Convergence criteria
    CC = 1 * 10**-5

    # Number of nodes in clay
    node_num = 10

    # Using available heads as proxy for missing
    proxyflag = 1

    # For each pumping scenario
    for index, pumpsheet in enumerate(pumpsheets):

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

        # Dictionary to store everything
        model_sub = {"wellnestlist": wellnestlist,
                     "all_results": all_results,
                     "sub_total": sub_total,
                     "subv_total": subv_total,
                     "ann_sub": ann_sub,
                     "tmin": tmin,
                     "tmax": tmax,
                     "Thick_data": Thick_data,
                     "Sske_data": Sske_data,
                     "Sskv_data": Sskv_data,
                     "K_data": K_data,
                     "pumping_scenario": scenarios[index],
                     "CC": CC,
                     "clay_nodes": node_num,
                     "proxyflag": proxyflag,
                     "mode": mode}

        # If saving
        if saving == 1:

            # Path to save models
            path = os.path.abspath("models")

            # Saving dict for this model
            afile = open(path + "\\Allnests_sub_" + scenarios[index] + ".pkl", "wb")
            pickle.dump(model_sub, afile)
            afile.close()

# if importing subsidence model results
else:

    # Path to import models
    path = os.path.abspath("models")
    scenarios = ["500", "250", "100", "500_250", "0"]
    for scenario in scenarios:

        # Reload object from file
        file2 = open(path + "\\Allnests_sub_" + scenario + ".pkl", "rb")
        model_sub = pickle.load(file2)
        file2.close()

        all_ann_subs.append(model_sub["ann_sub"])

# Plotting
# path to save figures
path = os.path.abspath("figures")

# %%###########################################################################
# Plots Results: Line graphs of cumulative sub forecast for whole time period
# For appendix
##############################################################################

bkk_sub_gw.bkk_plotting.sub_forecast(path, model_sub["wellnestlist"],
                                     all_ann_subs,
                                     save=1)

# %%###########################################################################
# Plots Results: Maps of cumulative sub forecast from new tmin and tmax
# For main paper
##############################################################################

tmin = "2020"
tmax = "2060"
bkk_sub_gw.bkk_plotting.sub_forecast_map(path, model_sub["wellnestlist"],
                                         all_ann_subs, tmin, tmax,
                                         save=1)

# %%###########################################################################
# Plots Results: Sensitivity Analysis, shown in appendix
##############################################################################

# Creating (0) or importing (1)
importing = 1

# If saving odels
saving = 0

# Sensitivity analysis
sens_modes = ["Sske_clay", "thick", "Sskv", "K", "Sske_sand"]

# Recommended looking at results from sensitivity analysis for only
# one well nest
# Well nest to run sensitivity analysis
wellnest_sens = ["LCBKK013"]

# For each sensitivity parameter set
for sens_mode in sens_modes:

    # Increasing by 10%
    coeff = .5
    num = 11  # Num of increases in percentage

    # Preallocation
    # All results from every sensitivity
    sens_results = []
    sens_sub = []
    sens_subv = []
    sens_ann = []

    # If creating results for first time
    if importing == 0:

        tmin = "1978"
        tmax = "2060"

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

                # If last sens, setting sand elastic storage to clay
                # which is typically one order of magnitude higher
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

            # Dictionary to store everything
            model_sub = {"wellnestlist": wellnest_sens,
                         "all_results": all_,
                         "sub_total": sub_,
                         "subv_total": subv_,
                         "ann_sub": ann_,
                         "sens_mode": sens_mode,
                         "tmin": tmin,
                         "tmax": tmax,
                         "Thick_data": Thick_data,
                         "Sske_data": Sske_data,
                         "Sskv_data": Sskv_data,
                         "K_data": K_data,
                         "pumping_scenario": psheet,
                         "CC": CC,
                         "clay_nodes": node_num,
                         "proxyflag": proxyflag}

            # If saving
            if saving == 1:

                # Path to save models
                path = os.path.abspath("models")

                # Saving dict for this model
                afile = open(path + "\\LCBKK013_sub_sens_" + str(round(coeff*100)) +
                             sens_mode + ".pkl", "wb")
                pickle.dump(model_sub, afile)
                afile.close()

            # Shifting parameter value
            coeff += .1

    # If importing model results
    else:

        # For each parameter increase
        for i in range(num):

            # Path to save models
            path = os.path.abspath("models")

            # Saving dict for this model
            afile = open(path + "\\LCBKK013_sub_sens_" + str(round(coeff*100)) +
                         sens_mode + ".pkl", "rb")
            model_sub = pickle.load(afile)
            afile.close()

            # Saving results
            sens_results.append(model_sub["all_results"])
            sens_sub.append(model_sub["sub_total"])
            sens_subv.append(model_sub["subv_total"])
            sens_ann.append(model_sub["ann_sub"])

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

# %%###########################################################################
# Finds outliers in subsidence observations
##############################################################################


def find_outliers_IQR(df):
    """
    finds outliers

    Parameters
    ----------
    df : dataframe
        dataframe of data.

    Returns
    -------
    outliers : dataframe
        outliers.

    """
    q1 = df.quantile(0.25)

    q3 = df.quantile(0.75)

    IQR = q3-q1

    outliers = df[((df < (q1-1.5*IQR)) | (df > (q3+1.5*IQR)))]

    return outliers


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

# For each well nest, finds outliers
IQRnum = 0
hampelnum = 0
for wellnest in wellnestlist:
    # BENCHMARK LEVELING
    # Subsidence plotting
    # Getting benchmark time series
    loc = os.path.join(os.path.abspath("inputs"),
                       "SurveyingLevels.xlsx")
    subdata = pd.read_excel(loc, sheet_name=wellnest + "_Leveling",
                            index_col=3)

    # SYNTHETIC DATA
    # loc = os.path.join(os.path.abspath("inputs"),
    #                    "synthetictruth_partial.xlsx")
    # subdata = pd.read_excel(loc, index_col=0)
    subdata = pd.DataFrame(subdata)
    subdata.index = pd.to_datetime(subdata.index)
    # Getting rid of benchmarks outside time period
    subdata = subdata[(subdata.Year <= 2020)]

    # Benchmarks should start at 0 at the first year.
    bench = subdata.loc[:, subdata.columns.str.contains("Land")]
    if (bench.iloc[0] != 0).any():
        bench.iloc[0] = 0

    # IMPORTANT INFO
    # For benchmark measurements, the first year is 0, the second year
    # is the compaction rate over that first year.
    # For implicit Calc, the first year has a compaction rate over that
    # year, so to shift benchmarks value to the previouse year to match
    # Index has the right years
    bench.index = bench.index.shift(-1, freq="D")
    bench["date"] = bench.index

    # Gets the last date of each year
    lastdate = bench.groupby(pd.DatetimeIndex(bench["date"]).year,
                             as_index=False).agg(
                                 {"date": max}).reset_index(drop=True)
    bench = bench.loc[lastdate.date]

    # Dataframe prep
    daterange = pd.date_range(dt.datetime(1978, 12, 31), periods=43,
                              freq="Y").tolist()
    df = pd.DataFrame(daterange, columns=["date"])

    # annual data in cm
    plot_data = df.merge(bench, left_on=df.date,
                         right_on=bench.index,
                         how="left")

    # plot_data = plot_data.drop(columns=["date_x", "date_y"])
    # Renaming for second merge
    plot_data = plot_data.rename(columns={"key_0": "key0"})

    plot_data = plot_data.dropna(axis=0)

    # OBSERVATION
    dobs = plot_data[plot_data.columns[
                     plot_data.columns.str.contains("Land")].item()]
    dobs = dobs[dobs != 0]
    dobs.index = plot_data.key0[dobs.index]

    # FINDS OUTLIERS
    IQRoutliers = find_outliers_IQR(-dobs)
    hampeloutliers = hampel(-dobs, window_size=3, n_sigma=3.5).outlier_indices

    print(wellnest)
    print("\n IQR Method")
    print("\nnumber of outliers: " + str(len(IQRoutliers)))

    if len(IQRoutliers) > 0:

        IQRnum += 1

    if len(hampeloutliers) > 0:

        hampelnum += 1

    print("\nmax outlier value: " + str(IQRoutliers.max()))

    print("\nmin outlier value: " + str(IQRoutliers.min()))

    print("\n Hampel Method")
    print("\nnumber of outliers: " + str(len(hampeloutliers)))

    print("\nmax outlier value: " + str(-dobs[hampeloutliers].max()))

    print("\nmin outlier value: " + str(-dobs[hampeloutliers].min()))

print("IQR number of well nests: " + str(IQRnum))
print("Hampel number of well nests: " + str(hampelnum))
