# ##############################################################################
"""Calculate subsidence in BKK at wellnests with 8 aquifers but simulates top four.

# Plotting figures for Groundwater Paper
# Article Title: Hybrid data-driven and physics-based modeling of ground-
water and subsidence with an application to Bangkok, Thailand

Jenny Soonthornrangsan 2023
TU Delft
"""
# ##############################################################################

###############################################################################
# import statements
###############################################################################

# Importing packages and libraries
import os
import pandas as pd
import numpy as np
import pastas as ps
import matplotlib.pyplot as plt
import datetime
from matplotlib.ticker import (AutoMinorLocator)

# Bangkok Subsidence Model Package
import bkk_sub_gw

# Importing script for pre-processing Thai GW data
import main_functions as mfs

# Changing current directory to locaiton of python script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# %%###########################################################################
# Plotting settings
###############################################################################

plt.rc("font", size=12)  # controls default text size
plt.rc("axes", titlesize=5)  # fontsize of the title
plt.rc("axes", labelsize=6)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=6)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=6)  # fontsize of the y tick labels
plt.rc("legend", fontsize=6)  # fontsize of the legend

# %% Pumping for four different sceniarios
# Plotting pumping
# Reading in data
sheet = "EstTotalPump_54-60_Int50"
full_path = os.path.join(os.path.abspath("inputs"), "BasinPumping.xlsx")
pump_50 = pd.read_excel(full_path, sheet_name=sheet)
sheet = "EstTotalPump_54-60_IntF25"
pump_25 = pd.read_excel(full_path, sheet_name=sheet)
sheet = "EstTotalPump_54-60_IntF100"
pump_100 = pd.read_excel(full_path, sheet_name=sheet)
sheet = "EstTotalPump_54-60_IntF50_25"
pump_50_25 = pd.read_excel(full_path, sheet_name=sheet)
sheet = "EstTotalPump_54-60_IntF0"
pump_0 = pd.read_excel(full_path, sheet_name=sheet)

# Plotting
plt.figure(figsize=(3.2, 2.2), dpi=300)
plt.plot(pump_50.Date, pump_50.Pump2, linewidth=1.5, label="500,000 m$^3$/day",
         color="hotpink")
plt.plot(pump_50.Date, pump_25.Pump2, linewidth=1.5, label="250,000 m$^3$/day",
         color="tab:orange")
plt.plot(pump_50.Date, pump_50_25.Pump2, linewidth=1.5,
         label="Delayed\n250,000 m$^3$/day",
         color="tab:green")
plt.plot(pump_50.Date, pump_100.Pump2, linewidth=1.5, label="1,000,000 m$^3$/day",
         color="tab:red")
plt.plot(pump_50.Date, pump_0.Pump2, linewidth=1.5, label="No Pumping",
         color="tab:purple")
plt.plot(pump_50.Date[:24472], pump_50.Pump2[:24472], linewidth=1.5, color="k",
         label="Observed Pumping")
plt.legend()
plt.grid(True, linestyle="dotted")
ax = plt.gca()
# ax.set_xticklabels(x.year)
plt.xlabel("Year")
plt.ylabel("Pumping Rate (m$^3$/day)")

# Saving
path = "figures"
fig_name = "1954-2060Pumping.png"
full_figpath = os.path.join(path, fig_name)
plt.savefig(full_figpath, dpi=300, bbox_inches="tight", format="png")

fig_name = "1954-2060Pumping.eps"
full_figpath = os.path.join(path, fig_name)
plt.savefig(full_figpath, dpi=300, bbox_inches="tight", format="eps")

# %% Pumping until 2020
# Plotting pumping
# Reading in data
sheet = "EstTotalPump_54-60_Int50"
full_path = os.path.join(os.path.abspath("inputs"), "BasinPumping.xlsx")
pump_2020 = pd.read_excel(full_path, sheet_name=sheet)

# Xticks
x = pd.date_range(start=pump_2020.Date[0],
                  end=pump_2020.Date[24472],
                  periods=8)

# Plotting
plt.figure(figsize=(3.2, 2.2), dpi=300)
plt.plot(pump_2020.Date[:24472], pump_2020.Pump2[:24472], linewidth=1.5)
plt.xticks(x)
ax = plt.gca()
ax.set_xticklabels(x.year)
ax.xaxis.set_minor_locator(AutoMinorLocator(3))
plt.grid(True, linestyle=(0, (1, 5)), which="minor")
plt.grid(True, linestyle="dotted", which="major")
plt.xlabel("Year")
plt.ylabel("Pumping Rate (m$^3$/day)")
# plt.title("Basin-Wide Pumping Estimates for Bangkok")

# Saving
path = "figures"
fig_name = "1954-2020Pumping.png"
full_figpath = os.path.join(path, fig_name)
plt.savefig(full_figpath, bbox_inches="tight", format="png")

fig_name = "1954-2020Pumping.eps"
full_figpath = os.path.join(path, fig_name)
plt.savefig(full_figpath, bbox_inches="tight", format="eps")

# %% Plotting one-day block response
modelpath = os.path.abspath("models")
# Model files
modelfiles = os.listdir(modelpath)
Wellnest_name = "LCBKK013"
well_name = "PD32"
wellmodel = [s for s in modelfiles
             if np.logical_and(Wellnest_name in s, well_name in s)][0]
model = ps.io.load(modelpath + "\\" + wellmodel)

# Block response
b = model.get_block_response("well")*1000
x = np.linspace(0, 2000, 2000)
y = np.append(0, b)

# Plotting
plt.figure(figsize=(3.2, 2.2), dpi=300)
plt.plot(x, y[:2000], linewidth=1.5)
plt.grid(True, linestyle="dotted")
plt.xlabel("Days")
plt.ylabel("Groundwater Head (m)")
# plt.title("Groundwater Response from \n1000 m$^3$/day of Pumping for One Day")

# Saving
path = "figures"
fig_name = "BlockResponseEx.png"
full_figpath = os.path.join(path, fig_name)
plt.savefig(full_figpath, bbox_inches="tight", format="png")

fig_name = "BlockResponseEx.eps"
full_figpath = os.path.join(path, fig_name)
plt.savefig(full_figpath, bbox_inches="tight", format="eps")
# %% Plotting Groundwater forecasts
modelpath = os.path.abspath("models")
# Model files
modelfiles = os.listdir(modelpath)
Wellnest_name = "LCBKK013"
well_name = "PD32"
wellmodel = [s for s in modelfiles
             if np.logical_and(Wellnest_name in s, well_name in s)][0]
model = ps.io.load(modelpath + "\\" + wellmodel)
pump_rfunc = ps.Gamma()
pumppath = os.path.join(os.path.abspath("inputs"), "BasinPumping.xlsx")
EstTotPump = pd.read_excel(full_path, sheet_name=sheet)
pumpsheet = "EstTotalPump_54-60_Int50"

# Original pumping scenario 500,000 m3/day
# Loading model and simulating based on new scenario
time_min = "1988"
time_max = "2060"
head50 = model.simulate(tmin=time_min, tmax=time_max)

# Pumping scenario 250,000 m3/day
# Loading model and simulating based on new scenario
optiparam = model.parameters["optimal"]
stdparam = model.parameters["stderr"]
model.del_stressmodel("well")
pumpsheet = "EstTotalPump_54-60_IntF25"
EstTotPump = pd.read_excel(pumppath, sheet_name=pumpsheet, index_col=0,
                           parse_dates=["Date"])
EstTotPump_ = ps.StressModel(EstTotPump.Pump, rfunc=pump_rfunc, name="well",
                             settings="well", up=False)
model.add_stressmodel(EstTotPump_)
model.parameters["optimal"] = optiparam
model.parameters["stderr"] = stdparam

head25 = model.simulate(tmin=time_min, tmax=time_max)

# Pumping scenario 1,000,000 m3/day
# Loading model and simulating based on new scenario
optiparam = model.parameters["optimal"]
stdparam = model.parameters["stderr"]
model.del_stressmodel("well")
pumpsheet = "EstTotalPump_54-60_IntF100"
EstTotPump = pd.read_excel(pumppath, sheet_name=pumpsheet, index_col=0,
                           parse_dates=["Date"])
EstTotPump_ = ps.StressModel(EstTotPump.Pump, rfunc=pump_rfunc, name="well",
                             settings="well", up=False)
model.add_stressmodel(EstTotPump_)
model.parameters["optimal"] = optiparam
model.parameters["stderr"] = stdparam

head100 = model.simulate(tmin=time_min, tmax=time_max)

# Pumping scenario 500,000 to 250,000 m3/day
# Loading model and simulating based on new scenario
optiparam = model.parameters["optimal"]
stdparam = model.parameters["stderr"]
model.del_stressmodel("well")
pumpsheet = "EstTotalPump_54-60_IntF50_25"
EstTotPump = pd.read_excel(pumppath, sheet_name=pumpsheet, index_col=0,
                           parse_dates=["Date"])
EstTotPump_ = ps.StressModel(EstTotPump.Pump, rfunc=pump_rfunc, name="well",
                             settings="well", up=False)
model.add_stressmodel(EstTotPump_)
model.parameters["optimal"] = optiparam
model.parameters["stderr"] = stdparam

head50_25 = model.simulate(tmin=time_min, tmax=time_max)

# Pumping scenario 0 m3/day
# Loading model and simulating based on new scenario
optiparam = model.parameters["optimal"]
stdparam = model.parameters["stderr"]
model.del_stressmodel("well")
pumpsheet = "EstTotalPump_54-60_IntF0"
EstTotPump = pd.read_excel(pumppath, sheet_name=pumpsheet, index_col=0,
                           parse_dates=["Date"])
EstTotPump_ = ps.StressModel(EstTotPump.Pump, rfunc=pump_rfunc, name="well",
                             settings="well", up=False)
model.add_stressmodel(EstTotPump_)
model.parameters["optimal"] = optiparam
model.parameters["stderr"] = stdparam

head0 = model.simulate(tmin=time_min, tmax=time_max)

# Plotting
plt.figure(figsize=(3.2, 2.2), dpi=300)
plt.rc("xtick", labelsize=6)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=6)  # fontsize of the y tick labels
plt.rc("legend", fontsize=5)
plt.plot(head50, linewidth=1.5, label="500,000 m$^3$/day",
         color="hotpink")
plt.plot(head25, linewidth=1.5, label="250,000 m$^3$/day",
         color="tab:orange")
plt.plot(head50_25, linewidth=1.5, label="Delayed\n250,000 m$^3$/day",
         color="tab:green")
plt.plot(head100, linewidth=1.5, label="1,000,000 m$^3$/day",
         color="tab:red")
plt.plot(head0, linewidth=1.5, label="No Pumping",
         color="tab:purple")
plt.plot(head0[:12054], linewidth=1.5, color="black",
         label="Observed Pumping")
plt.grid(True, linestyle="dotted")
plt.legend()
plt.xlabel("Years")
plt.ylabel("Groundwater Head (m)")
# plt.title("LCBKK013 PD32\nGroundwater Forecasts\nBased on Pumping Scenarios")

# saving
path = "figures"
fig_name = "GWForecasts_LCBKK013PD32.png"
full_figpath = os.path.join(path, fig_name)
plt.savefig(full_figpath, bbox_inches="tight", format="png")

fig_name = "GWForecasts_LCBKK013PD32.eps"
full_figpath = os.path.join(path, fig_name)
plt.savefig(full_figpath, bbox_inches="tight", format="eps")


# %% Data availability
# Pumping vs groundwater vs levelling measurements

# Plotting pumping
# Reading in data
sheet = "EstTotalPump_54-60_Int50"
pumppath = "inputs\\BasinPumping.xlsx"
pump_2020 = pd.read_excel(pumppath, sheet_name=sheet)

# Xticks
x = pd.date_range(start=pump_2020.Date[0],
                  end=pump_2020.Date[19733],
                  periods=8)

# Plotting
fig, axs = plt.subplots(3, sharex=True, figsize=(6.75, 3.38), dpi=300)
axs[0].plot(pump_2020.Date[:19733], pump_2020.Pump2[:19733], linewidth=1.5)
axs[0].set_ylabel("Pumping Rate\n(m$^3$/day)")
axs[0].set_title("Basin-Wide Pumping Estimates for Bangkok",
                 fontsize=7)
axs[0].grid(True, linestyle="dotted")
# Plottign groundwater
# Reading in groundwater data
Wellnest_name = "LCBKK013"
well_name = "PD32"
well_path = "inputs\\"
full_path = os.path.join(well_path, Wellnest_name + ".xlsx")
data = pd.read_excel(full_path, skiprows=3)
all_head_data, gw_well_head = mfs.GW_Data_Process(data, well_name)

# CORRECTING GW HEAD DATA TO LAND SURFACE (COASTAL DEM 2.1)
landsurf_path = os.path.join(well_path,
                             "LandSurfElev_GWWellLocs.xlsx")

# Each well nest has its own Ss and K sheet
landsurf_data = pd.read_excel(landsurf_path,
                              sheet_name="2.1",
                              usecols="C:F",
                              index_col=0)

gw_well_head.Head += (landsurf_data.RASTERVALU.loc[Wellnest_name])
# Adding years and annual average heads
gw_well_head["year"] = gw_well_head.index.year
axs[1].plot(gw_well_head.index, gw_well_head.Head, color="k",
            linewidth=1.5)
axs[1].set_xlim(([datetime.date(1954, 1, 1), datetime.date(2020, 12, 31)]))
axs[1].set_ylabel("Head (m)")
axs[1].set_title("Groundwater Levels for Well PD32 in Well Nest LCBKK013",
                 fontsize=7)
axs[1].grid(True, linestyle="dotted")
loc = os.path.join(os.path.abspath("inputs"), "SurveyingLevels.xlsx")

subdata = pd.read_excel(loc, sheet_name=Wellnest_name+"_Leveling",
                        index_col=3)
subdata = pd.DataFrame(subdata)
subdata.index = pd.to_datetime(subdata.index)

# Getting rid of benchmarks outside time period
subdata = subdata[(subdata.Year <= 2020)]

# Benchmarks should start at 0 at the first year.
bench = subdata.loc[:, subdata.columns.str.contains("Land")]
bench = bench.fillna(0)

if (bench.iloc[0] != 0).any():
    bench.iloc[0] = 0

# IMPORTANT INFO
# For benchmark measurements, the first year is 0, the second year is
# the compaction rate over that first year.
# For implicit Calc, the first year has a compaction rate over that
# year, so to shift benchmarks value to the previouse year to match
# Index has the right years
bench.index = bench.index.shift(-1, freq="D")
bench["date"] = bench.index

# Gets the last date of each year
lastdate = bench.groupby(pd.DatetimeIndex(bench["date"]).year,
                         as_index=False).agg({"date": max}).reset_index(drop=True)
bench = bench.loc[lastdate.date]

leveling = bench[
    bench.columns[
        bench.columns.str.contains("Land")].item()]

leveling[leveling == 0] = np.nan
axs[2].plot(leveling, "o", color="orange", linewidth=1.5,
            markersize=3)
axs[2].set_xlim(([datetime.date(1954, 1, 1), datetime.date(2020, 12, 31)]))
axs[2].set_ylabel("Annual Rate\n(cm/year)")
axs[2].set_title(
    "Annual Land Subsidence Rates from Benchmark Leveling Station 5503",
    fontsize=7)
plt.tight_layout()
plt.rc("font", size=10)  # controls default text size
axs[2].grid(True, linestyle="dotted")

# Saving
path = "figures"
fig_name = "DataAvailability.eps"
full_figpath = os.path.join(path, fig_name)
plt.savefig(full_figpath, bbox_inches="tight", format="eps")

fig_name = "DataAvailability.png"
full_figpath = os.path.join(path, fig_name)
plt.savefig(full_figpath, bbox_inches="tight", format="png")

# %% Plotting groundwater well locations

path = os.path.abspath("figures")
bkk_sub_gw.bkk_plotting.gwlocs_map(path, save=1)
