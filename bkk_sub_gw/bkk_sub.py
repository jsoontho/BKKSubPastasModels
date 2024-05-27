# ##############################################################################
"""Calculate subsidence in BKK at wellnests with 8 aquifers but simulates top four.

BK, PD, NL, NB
all are confined and overlain by clay layer
Implicit method according to USGS SUB package Hoffman report pg. 14

Article Title: Hybrid data-driven, physics-based modeling of ground-
water and subsidence with application to Bangkok, Thailand

Jenny Soonthornrangsan 2023
TU Delft

"""
# ##############################################################################

###############################################################################
# import statements
###############################################################################

import re
import os
import pandas as pd
import numpy as np
import scipy.linalg as lin
import sys
import functools
import pastas as ps

# Importing script for pre-processing Thai GW data
import main_functions as mfs

pd.options.mode.chained_assignment = None  # default='warn'

# %%###########################################################################
# Preprocessing GW well nest data
###############################################################################

def bkk_wellnest_preproc(wellnestname, tmin, tmax, proxyflag):
    """Take well nest name, load data, and clean it up.

    Returns data within tmin and tmax
    By cleaning, gets rid of thai characters, interpolates dates and head
    Keeps matching dates and data only between wells in the nest
    wellnestname-- (str) name of well nest
    tmin, tmax -- (str) minimum and maximum year, if min year = 1900 but
    data starts at 1960, will return data from 1960 onwards
    proxyflag - 1 if using available heads as proxy for missing

    Returns: well_data_dates - dataframe with only matching dates and data
    between wells
    well_data - dataframe with matching dates, some wells have missing data
    """
    # Reading in GW data
    # Path to GW data
    try:
        full_path = os.path.join(os.path.abspath("inputs"), wellnestname + ".xlsx")
        data = pd.read_excel(full_path, skiprows=3)

    # If well nest does not exist
    except:
        raise ValueError("\nWell nest or file for well nest does not exist.")

    # List of wells in well nest
    welllist = data.columns[-(len(data.columns) - 2):]

    # Reorder well list to shallow to deep aquifers
    # BK, PD, NL, NB
    welllist = [x for y in ["BK", "PD", "NL", "NB"] for x in welllist
                if y in x]

    # Returns all data, and specific well data if specified.
    # GW Head not DTW
    all_head_data, gw_well_head = mfs.GW_Data_Process(data)
    all_head_data.index = all_head_data.EngDate

    # Stores interpolated data
    interp_welldata = []

    # For each well in welllist
    for i in welllist:

        # Gets rid of NA, then resamples daily, then does cubic interpolation
        # Interpolating 'inside'
        interp_welldata.append(
            all_head_data.loc[:, i].dropna().
            resample("D").interpolate("linear"))

    lenlist = len(welllist)

    # If using available heads as proxy for missing heads
    if proxyflag == 1:

        well_data = []

        # For those missing wells, checks which well is missing
        if lenlist < 4:

            # If only three files
            # If first file in sorted list is PD, missing BK
            if lenlist == 3:

                if not any("BK" in substring for substring in welllist):

                    print("\nPROXY\n")
                    # Using PD as BK Proxy
                    temp = interp_welldata[0]
                    temp = temp.rename("Proxy BK")
                    well_data.append(temp)

                    # Adding rest of the wells
                    [well_data.append(i) for i in interp_welldata]

            # If only two files
            # If first file in sorted list is PD and next is NB
            # missing BK and NL
            elif lenlist == 2:

                if np.logical_and(not any("BK" in substring
                                          for substring in welllist),
                                  not any("NL" in substring
                                          for substring in welllist)):

                    # Using PD as BK Proxy
                    temp = interp_welldata[0]
                    temp = temp.rename("Proxy BK")
                    well_data.append(temp)

                    # PD
                    temp = interp_welldata[0]
                    well_data.append(temp)

                    # Using NB as NL Proxy
                    temp = interp_welldata[1]
                    temp = temp.rename("Proxy NL")
                    well_data.append(temp)

                    # NB
                    temp = interp_welldata[1]
                    well_data.append(temp)

            # If only two files
            # If first file in sorted list is PD, next and NL
            # missing BK and NB

                elif np.logical_and(
                        not any("BK" in substring for substring in welllist),
                        not any("NB" in substring for substring in welllist)):

                    # Using PD as BK Proxy
                    temp = interp_welldata[0]
                    temp = temp.rename("Proxy BK")
                    well_data.append(temp)

                    # PD
                    temp = interp_welldata[0]
                    well_data.append(temp)

                    # NL
                    temp = interp_welldata[1]
                    well_data.append(temp)

                    # Using NL as NB proxy
                    temp = interp_welldata[1]
                    temp = temp.rename("Proxy NB")
                    well_data.append(temp)

            # If only two files
            # If first file in sorted list is NL, next and NB
            # missing BK and PD

                elif np.logical_and(
                        not any("BK" in substring for substring in welllist),
                        not any("PD" in substring for substring in welllist)):

                    # Using NL as BK Proxy
                    temp = interp_welldata[0]
                    temp = temp.rename("Proxy BK")
                    well_data.append(temp)

                    # Using NL as PD Proxy
                    temp = interp_welldata[0]
                    temp = temp.rename("Proxy PD")
                    well_data.append(temp)

                    # NL
                    temp = interp_welldata[0]
                    well_data.append(temp)

                    # Using NL as NB proxy
                    temp = interp_welldata[1]
                    well_data.append(temp)

            # If only 1 file
            # missing others
            if lenlist == 1:

                # If only has BK
                if any("BK" in substring for substring in welllist):

                    # BK
                    temp = interp_welldata[0]
                    well_data.append(temp)

                    # Using BK as PD Proxy
                    temp = interp_welldata[0]
                    temp = temp.rename("Proxy PD")
                    well_data.append(temp)

                    # Using BK as NL Proxy
                    temp = interp_welldata[0]
                    temp = temp.rename("Proxy NL")
                    well_data.append(temp)

                    # Using BK as NB Proxy
                    temp = interp_welldata[0]
                    temp = temp.rename("Proxy NB")
                    well_data.append(temp)

                # If only has PD
                elif any("PD" in substring for substring in welllist):

                    # Using PD as BK Proxy
                    temp = interp_welldata[0]
                    temp = temp.rename("Proxy BK")
                    well_data.append(temp)

                    # PD
                    temp = interp_welldata[0]
                    well_data.append(temp)

                    # Using PD as NL Proxy
                    temp = interp_welldata[0]
                    temp = temp.rename("Proxy NL")
                    well_data.append(temp)

                    # Using PD as NB Proxy
                    temp = interp_welldata[0]
                    temp = temp.rename("Proxy NB")
                    well_data.append(temp)

                # If only has NL
                elif any("NL" in substring for substring in welllist):

                    # Using NL as BK Proxy
                    temp = interp_welldata[0]
                    temp = temp.rename("Proxy BK")
                    well_data.append(temp)

                    # Using NL as PD Proxy
                    temp = interp_welldata[0]
                    temp = temp.rename("Proxy PD")
                    well_data.append(temp)

                    # Using NL
                    temp = interp_welldata[0]
                    well_data.append(temp)

                    # Using NL as NB Proxy
                    temp = interp_welldata[0]
                    temp = temp.rename("Proxy NB")
                    well_data.append(temp)

                # If only has NB
                elif any("NB" in substring for substring in welllist):

                    # Using NB as BK Proxy
                    temp = interp_welldata[0]
                    temp = temp.rename("Proxy BK")
                    well_data.append(temp)

                    # Using NB as PD Proxy
                    temp = interp_welldata[0]
                    temp = temp.rename("Proxy PD")
                    well_data.append(temp)

                    # Using NB as NL Proxy
                    temp = interp_welldata[0]
                    temp = temp.rename("Proxy NL")
                    well_data.append(temp)

                    # NB
                    temp = interp_welldata[0]
                    well_data.append(temp)

        # No missing wells
        else:

            well_data = interp_welldata

    # If using only available heads
    else:

        well_data = interp_welldata

        # Needs all four wells if proxyflag is not on
        if lenlist < 4:

            sys.exit("Needs all four wells if proxyflag is not on")

    # Well data with matching dates only
    well_data_dates = functools.reduce(lambda left, right:
                                       pd.merge(left, right, on=["EngDate"],
                                                how="inner"), well_data)
    well_data_dates = well_data_dates[(well_data_dates.index.year >= int(tmin)) &
                                      well_data_dates.index.year <= int(tmax)]

    # All well data
    all_well_data = functools.reduce(lambda left, right:
                                     pd.merge(left, right, on=["EngDate"],
                                              how="outer", sort=True),
                                     well_data)

    return well_data_dates, all_well_data


# %%###########################################################################
# Groundwater model for clay: Using Matrixes to solve for head
###############################################################################

class SolveFDM:
    """Solve for head for time step n using the finite difference implicit method.

    Solving for x (head) in Ax = b (See USGS SUB report by Hoffman pg 14)
    """

    def __init__(self, Nz, n, dz, Kv, Sskv, Sske,
                 dt, precon, CC, toplay):

        self.Nz = Nz  # Number of nodes
        self.n = n  # Current time step
        self.dz = dz  # Cell diff
        self.Kv = Kv  # Vertical hydraulic conductivity
        self.Sskv = Sskv  # Specific storage (inelastic)
        self.Sske = Sske  # Specific storage (elastic)
        self.dt = dt  # Time diff
        self.precon = precon  # Preconsolidated head
        self.CC = CC  # Convergence criteria
        self.toplay = toplay  # If top layer or not

    # Checking if elastic or inelastic for each clay node
    def ElasticInelastic(self, h_matr):
        """Check if elastic or inelastic for each clay node.

        Input: h_matrx - head matrix
        Output:
        Ss - array the size of precon with either Sskv or Sske for each node
        precons - array with the preconsolidated head for each node
        """
        # Creating Ss array
        Ss = np.zeros(np.shape(self.precon))

        # Creating array for updated preconsolidated head
        precons = self.precon.copy()

        # For each inner node, ignoring aquifer nodes
        for i in range(1, self.Nz+1):

            # If it is the first time step
            if self.n == 1:

                # If current head is less than or equal (INELASTIC)
                if h_matr[i, self.n-1] <= self.precon[i]:

                    # Saves inelastic storage term for this node
                    Ss[i] = self.Sskv

                    # Sets new preconsolidation head
                    precons[i] = h_matr[i, self.n-1]

                # ELASTIC
                else:

                    # Saves elastic storage term for this node
                    Ss[i] = self.Sske

            # All other time steps
            else:

                # Difference in head for the previouse two time steps
                dh = h_matr[i, self.n-1] - h_matr[i, self.n-2]

                # If current head is less than or equal (INELASTIC) and if
                # slope is negative
                if np.logical_and(h_matr[i, self.n-1] <= self.precon[i], dh < 0):

                    # Inelastic storage term saved
                    Ss[i] = self.Sskv

                    # Sets new preconsolidation head
                    precons[i] = h_matr[i, self.n-1]

                # ELASTIC
                else:
                    Ss[i] = self.Sske

        # Returning
        return Ss, precons

    # Building A matrix
    def buildCoeffMatrix(self, h_matr):
        """Build A matrix.

        Input: h_matrix - head matrix
        Output:
        A - A matrix
        Ss - array with either elastic/inelastic storage term for each node
        precon - updated preconsolidated head for each node
        """
        Ss, precon = self.ElasticInelastic(h_matr)

        # Preallocation
        Adiag_val = np.ones(self.Nz)

        # For each main diagonal except the first and last inner node
        # IMPORTANT: Ss array includes cell for aquifer on top and bottom
        # Diag only for inner nodes -> thus, inner node 2 has index of 1 in the
        # diagonal while index of 2 in the Ss (i+1)
        for i in range(1, self.Nz-1):
            Adiag_val[i] = (-2 * self.Kv / self.dz) - (self.dz / self.dt * Ss[i+1])

        # First value and last value of the main diagonal
        # Inner nodes that border the aquifer
        # First inner node near top aquifer
        # If not top clay layer, the top is an aquifer
        if self.toplay is False:
            Adiag_val[0] = (-3 * self.Kv / self.dz) - (self.dz / self.dt * Ss[1])

        # If top clay layer, the top is a noflow boundary
        else:
            Adiag_val[0] = -(self.Kv / self.dz) - (self.dz / (self.dt) * Ss[1])

        # Last inner node near bottom aquifer
        Adiag_val[-1] = (-3 * self.Kv / self.dz) - (self.dz / self.dt * Ss[-2])

        # Creating A matrix
        Aupper = np.diag(np.ones(self.Nz-1) * self.Kv / self.dz, 1)  # Upper diag
        Alower = np.diag(np.ones(self.Nz-1) * self.Kv / self.dz, -1)  # Lower diag
        Adiag = np.diag(Adiag_val)  # Main diagonal
        A = Alower + Aupper + Adiag

        # Returning
        return A, Ss, precon

    # Building b matrix
    def buildRHSVector(self, h_matr, Ss, precon):
        """Right hand side vector (b).

        Input:
        h_matr - head matrix
        Ss - array of either elastic or inelastic storage for each node
        precon - array of updated preconsolidated head for each node
        """
        # Preallocation
        b = np.ones(self.Nz)

        # For each inner node that is not the first/last
        # IMPORTANT: Ss/h_matr/precon array includes cell for aquifer on top
        # and bottom; b only for inner nodes -> thus, inner node 2 has index
        # of 1 in b while index of 2 in the other arrays (i+1)
        for i in range(1, self.Nz-1):
            b[i] = (self.dz/self.dt) * (-Ss[i+1] * self.precon[i+1] +
                                        self.Sske * (self.precon[i+1] -
                                                     h_matr[i+1, self.n-1]))

        # If not top clay layer, the top is an aquifer
        if self.toplay is False:
            # First inner node near top aquifer
            b[0] = (self.dz/self.dt) * (-Ss[1] * self.precon[1] +
                                        self.Sske * (self.precon[1] -
                                                     h_matr[1, self.n-1])) - \
                2 * self.Kv / self.dz * h_matr[0, self.n]

        # If top clay layer, the top is a noflow boundary
        else:
            # First inner node near top aquifer
            b[0] = (self.dz/self.dt) * (-Ss[1] * self.precon[1] +
                                        self.Sske * (self.precon[1] -
                                                     h_matr[1, self.n-1]))

        # Last inner node near bottom aquifer
        b[-1] = (self.dz/self.dt) * (-Ss[-2] * self.precon[-2] +
                                     self.Sske * (self.precon[-2] -
                                                  h_matr[-2, self.n-1])) - \
            2 * self.Kv / self.dz * h_matr[-1, self.n]

        # Returning
        return b

    def solveLinearSystem(self, A, b):
        """Solve linear system of matrices."""
        h = lin.solve(A, b)
        return h

    # Iterates through until all cells meet the convergence criteria for a time
    # step n
    def iterate(self, h_matr, precons_head):
        """Iterate until all cells meet the convergence criteria for a timestep n.

        Input:
        h_matr - head matrix
        precons_head - current preconsolidated head for each node at the
        start of the time step
        Output:
        h_matr - head matrix updated with new heads in n time step after
        iterating
        precons_head - updated preconsolidated head for each node at the end
        of the time step
        """
        # Preallocation for the head diff in each cell
        Cell_change = np.ones(self.Nz)

        # Sets the starting new heads
        h_new = h_matr[1:self.Nz+1, self.n].copy()

        # While head diff of cells is greater than convergence criteria
        # Iterates
        while np.sum(Cell_change > self.CC) > 0:

            # Remembers old head
            old_head = h_new.copy()

            # Creates new class with updated precons_head
            fdm = SolveFDM(self.Nz, self.n, self.dz, self.Kv, self.Sskv, self.Sske,
                           self.dt, precons_head, self.CC,
                           self.toplay)

            # Builds A matrix and updates Ss and preconsolidated head
            A, Ss, precons_head = fdm.buildCoeffMatrix(h_matr)

            # Builds right hand side array
            b = fdm.buildRHSVector(h_matr, Ss, precons_head)

            # Solves for head using A and RHS matrix b
            h_new = fdm.solveLinearSystem(A, b)

            # Checks for the difference between iterations
            Cell_change = np.abs(np.subtract(h_new, old_head))

        # Adds the aquifer heads at the top and bottom
        h_new = np.insert(h_new, 0, h_matr[0, self.n])
        h_new = np.append(h_new, h_matr[-1, self.n])

        # Saves new head array in the current time step in the head matrix
        h_matr[:, self.n] = h_new

        # Returning updated head matrix and preconsolidated head after iterate
        return h_matr, precons_head


# %%###########################################################################
# Subsidence model: calculates compaction for each layer
##############################################################################


# Calculates compaction based on groundwater levels in aquifer above and
# below a clay layer
def calc_deformation(timet, headt, headb, Kv, Sskv, Sske, Sske_sandt,
                     Sske_sandb, claythick, nclay, sandthickt, sandthickb,
                     Nt, CC, Nz=None, ic=None):
    """Calculate deformation for a single clay layer of user defined thickness.

    Use whatever units for time and length as desired, but they need to stay
    consistent
    Inputs:
    timet - a vector of same lenght as head with the times that head
    measurements are taken. Numeric (years or days, typically)
    headt - a vector of same length as time. Head of top aquifer
    headb - a vector of same length as time. Head of bottom aquifer
    Kv - vertical hydraulic conductivity
    Sske - Skeletal specific storage (elastic)
    Sskv - skeletalt specific storage (inelastic)
    Sske_sandt - Skeletal specific storage (elastic) of aq on top
    Sske_sandb - Skeletal specific storage (elastic) of aq on bottom
    claythick - thickness of single clay layer modeled
    nclay - number of clay layers
    sandthickt -  thickness of sand in top aquifer
    sandthickb -  thickness of sand in bottom aquifer
    Nz - number of layers in z direction, within the clay layer modeled.
    Nt - number of time steps
    CC - convergence criteria
    ic - if providing initial condition of clay, ndarray given

    Outputs:
    t - interpolated time
    deformation - cumulative sum of deformation of total clay layer (m)
    boundaryt - interpolated head at the top boundary
    boundaryb - interpolated head at the bottom boundary
    deformation_v - cumulative sum of inelastic deformation of total clay (m)
    h - aquifer heads row 0 and -1, rest are clay nodes head
    """
    # Storage coefficients of aquifers top and bottom
    Ske_sandt = Sske_sandt * sandthickt
    Ske_sandb = Sske_sandb * sandthickb

    # Interpolated time
    # The first time step (0) doesn't count so needs Nt + 1
    t = np.linspace(timet[0], timet[-1], int(Nt+1))

    # Interpolated head at the boundaries (aquifers)
    # Time and head have the same time steps. Interpolating head for t
    # Has the same number of steps as t
    # If not the top clay layer with no top aquifer
    if isinstance(headt, pd.Series):

        # Interpolating top and bottom aquifer heads
        boundaryt = np.interp(t, timet, headt)  # Linear
        boundaryb = np.interp(t, timet, headb)  # Linear

        # Initial conditions of head grid
        if isinstance(ic, np.ndarray):

            # Setting the head matrix to the initial condition
            h = np.tile(ic, (Nt+1, 1))
            h = h.transpose()

        h[0, :] = boundaryt
        h[-1, :] = boundaryb

        # It is not the top clay layer
        toplay = False

    # If top clay layer
    else:
        boundaryb = np.interp(t, timet, headb)  # Linear
        boundaryt = np.zeros(np.shape(boundaryb))

        # Initial conditions of head grid
        if isinstance(ic, np.ndarray):

            # Setting the head matrix to the initial condition
            h = np.tile(ic, (Nt+1, 1))
            h = h.transpose()

        else:

            # Initial conditions of head grid
            h = (headb[0])*np.ones((Nz+2, Nt+1))

        h[-1, :] = boundaryb

        # It is the top clay layer
        toplay = True

    # Initial precons head made of initial head in each node
    # Preconsolidated head set to head from first time step
    precons_head = h[:, 0].copy()

    # Preallocation for total/inelastic deformation
    deformation = np.zeros(np.shape(h))
    deformation_v = np.zeros(np.shape(h))

    # Length of z
    dz = claythick / (Nz)

    # For each time step
    # Starting at 1, because 0 doesn't count as a time step
    for n in range(1, Nt+1):

        # Difference in time
        dt2 = t[n] - t[n-1]

        # Finite difference implicit method solving for head at current time
        # step. Uses matrix. Iterative because Sskv and Sske can change
        fdm = SolveFDM(Nz, n, dz, Kv, Sskv, Sske, dt2,
                       precons_head, CC, toplay=toplay)
        h, precons_head = fdm.iterate(h, precons_head)

        # New head that is calculated is already saved to head matrix
        h_new = h[:, n]

        # Compute compaction
        for i in range(1, Nz+1):

            # Diff between new and old
            dh = (h_new[i] - h[i, n-1])

            # If head drops below preconsolidation head and slope is neg
            # INELASTIC
            if np.logical_and(h_new[i] <= precons_head[i], dh < 0):

                # Calculating deformation
                defm = dh * Sskv * dz

                # Adds new deformation to the min from before for this
                # node from time steps before
                deformation_v[i, n] = defm + np.min(deformation_v[i, 0:(n)])

            # ELASTIC
            else:

                # Calculating deformation
                defm = dh * Sske * dz

                # Next time step deformation equals this current time step
                deformation_v[i, n] = deformation_v[i, n-1]

            # Total deformation updated
            deformation[i, n] = defm + deformation[i, n-1]

    # Deformation multipled by number of clay layers
    deformation = np.sum(deformation, axis=0) * nclay
    deformation_v = np.sum(deformation_v, axis=0) * nclay

    # If not the top clay layer
    if isinstance(headt, pd.Series):

        # Interpolated head minus initial
        boundary0_t = boundaryt - boundaryt[0]

    # If the top clay layer
    else:
        boundary0_t = 0

        h[0, :] = h[1, :]

        # Top row equals the second row
        boundaryt = h[1, :]

    # Interpolated head minus initial
    boundary0_b = boundaryb - boundaryb[0]

    # Sand deformation (top and bottom aquifer)
    # All elastic
    sanddef = boundary0_t * Ske_sandt + boundary0_b * Ske_sandb

    # If adding deformation from sand
    deformation = deformation + sanddef

    # Returning
    return (t, deformation, boundaryt, boundaryb, deformation_v, h)


# %%###########################################################################
# Loads Pastas models
##############################################################################

def load_Pastas_models(Pastasfiles, model_path):
    """ Loads Pastas models and saves models for use later (so that models can be
    loaded only once)

    Pastasfiles- list of Pastas file names to load
    model_path - path of where files are located

    Return:
        models - list of Pastas models
        well_names - list of well names (str)
        modelinitparam - initial parameters
    """
    # Number of files
    num_models = len(Pastasfiles)

    # saves models
    models = []

    # saves well names for each model
    well_names = []

    # Pastas optimized parameters saved
    pastas_optparam = []

    # For each model
    for num in range(num_models):

        # Loads model
        model = ps.io.load(model_path + "/" + Pastasfiles[num])
        s = Pastasfiles[num]
        result = re.search("_(.*)_GW", s)  # Gets well name

        # Saving optimized parameters
        pastas_optparam.append(model.parameters["optimal"])

        # Saves
        models.append(model)
        well_names.append(result.group(1))

    return models, well_names, np.array(pastas_optparam)


# Future simulating Pastas with different pumping scenarios
def pastas_setparam(model, pump_series=None, pump_path=None, pump_sheet=None,
                    initoptiparam=None, well_name=None):
    """Pastas model that is already created.

    pump_series - if pump series given already
    pump_path - path to pumping excel sheet
    pump_sheet - sheet of specific pumping scenario
    initoptiparam - optimal parameters provided or not
    well_name - name of well
    """

    model.del_noisemodel()
    stdparam = model.parameters["stderr"]

    # Saves optimal parameters and SD
    if initoptiparam is None:

        optiparam = model.parameters["optimal"]

        # If no pump series provided
        if pump_series is None:

            # If pump path provided
            if pump_path is not None:
                model.del_stressmodel("well")  # Deletes previous pumping

                # Adds new pumping
                EstTotPump = pd.read_excel(pump_path,
                                           sheet_name=pump_sheet, index_col=0,
                                           parse_dates=["Date"])

                EstTotPump_ = ps.StressModel(EstTotPump.Pump,
                                             rfunc=ps.Gamma(), name="well",
                                             settings="well", up=False)
                model.add_stressmodel(EstTotPump_, replace=True)

        # If pumping series provided
        else:

            model.del_stressmodel("well")  # Deletes previous pumping

            EstTotPump_ = ps.StressModel(pump_series, rfunc=ps.Gamma(), name="well",
                                         settings="well", up=False)
            model.add_stressmodel(EstTotPump_)

        model.parameters["optimal"] = optiparam
        model.parameters["initial"] = optiparam
        model.parameters["stderr"] = stdparam

    else:

        # If no pump series provided
        if pump_series is None:

            # If pump path provided
            if pump_path is not None:

                model.del_stressmodel("well")  # Deletes previous pumping

                # Adds new pumping
                EstTotPump = pd.read_excel(pump_path,
                                           sheet_name=pump_sheet, index_col=0,
                                           parse_dates=["Date"])
                EstTotPump_ = ps.StressModel(EstTotPump.Pump,
                                             rfunc=ps.Gamma(), name="well",
                                             settings="well", up=False)
                model.add_stressmodel(EstTotPump_)

        # If pumping series provided
        else:

            model.del_stressmodel("well")  # Deletes previous pumping

            EstTotPump_ = ps.StressModel(pump_series, rfunc=ps.Gamma(), name="well",
                                         settings="well", up=False)

            model.add_stressmodel(EstTotPump_)

        model.parameters["initial"] = initoptiparam.values
        model.parameters["optimal"] = initoptiparam
        model.parameters["stderr"] = stdparam

    # Returns model
    return model


def load_Pastas(Pastasfiles, lenfiles, proxyflag, models, well_names,
                model_path, pumpflag, tmin, tmax, pump_series=None,
                pump_path=None, pump_sheet=None, initoptiparam=None):
    """Loads Pastas models

    Pastasfiles - list of Pastas file names
    models - lsit of Pastas model instances
    well_names - list of well names (str)
    lenfiles - how many files there are
    proxyflag - 1 if using available heads as proxy for missing heads
    pump_path - path to pumping excel sheet
    pump_sheet - sheet of specific pumping scenario
    model_path - path to python models
    pumpflag - 1 if changing pumping scenario for Pastas
    tmin, tmax - (str) minimum and maximum year to calculate sub
    initoptiparam - optimal parameters provided

    Returns
    well_data_dates - Well data with matching dates only
    all_well4_data - all well data no matter the date
    """
    # well_data - where to save the data after loading Pasta files
    well_data = []

    # If using avaialbe heads as proxy for missing heads
    if proxyflag == 1:

        # For those missing wells, checks which well is missing
        if lenfiles < 4:

            # If only three files
            # If first file in sorted list is PD, missing BK
            if lenfiles == 3:

                # For each well model
                for num_model in range(len(well_names)):

                    # Loads model, PD as proxy for BK
                    model = models[num_model]

                    curr_well = well_names[num_model]

                    # If changing pumping scenario
                    if pumpflag == 1:

                        # If providing optimal parameters from ESMDA
                        if initoptiparam is None:

                            # If pumping time series given
                            if pump_series is not None:
                                model = pastas_setparam(model,
                                                        well_name=curr_well,
                                                        pump_series=pump_series[
                                                            curr_well])

                            # If path and sheet given instead
                            elif pump_path is not None:
                                # Updating model with new pumping scenario
                                model = pastas_setparam(model, pump_path=pump_path,
                                                        pump_sheet=pump_sheet)

                        else:

                            # If pumping time series given
                            if pump_path is None:
                                if pump_series is not None:
                                    model = pastas_setparam(model,
                                                            initoptiparam=initoptiparam.
                                                            loc[
                                                                curr_well],
                                                            pump_series=pump_series[
                                                                curr_well],
                                                            well_name=curr_well,)

                                else:
                                    model = pastas_setparam(model,
                                                            initoptiparam=initoptiparam.
                                                            loc[
                                                                curr_well],
                                                            well_name=curr_well,)

                            # If path and sheet given instead
                            else:

                                # Updating model with new pumping scenario
                                model = pastas_setparam(model,
                                                        initoptiparam=initoptiparam.
                                                        loc[
                                                            curr_well],
                                                        well_name=curr_well,
                                                        pump_path=pump_path,
                                                        pump_sheet=pump_sheet)

                    # If list starts with PD, means BK is missing
                    if "_PD" in Pastasfiles[0]:

                        # Identifies missing well and index
                        missing = "BK"

                        temp = model.simulate(tmin="1950", tmax=tmax,
                                              warmup=365*30, return_warmup=False)
                        temp = temp.rename("Proxy BK")  # Renames column
                        well_data.append(temp)  # Saves data

                        # Loads model, PD
                        temp = temp.rename(curr_well)  # Renames column
                        well_data.append(temp)  # Saves data

                    # Adds NL and NB simulations
                    else:

                        temp = model.simulate(tmin="1950", tmax=tmax,
                                              warmup=365*30, return_warmup=False)
                        temp = temp.rename(curr_well)  # Renames col
                        well_data.append(temp)  # Saves data

            # If only two files
            # If first file in sorted list is PD and next is NB
            # missing BK and NL
            if lenfiles == 2:

                # For each well model
                for num_model in range(len(well_names)):

                    # Loads model, PD as proxy for BK
                    model = models[num_model]

                    curr_well = well_names[num_model]

                    # If changing pumping scenario
                    if pumpflag == 1:

                        # If providing optimal parameters from ESMDA
                        if initoptiparam is None:

                            # If pumping time series given
                            if pump_series is not None:
                                model = pastas_setparam(model,
                                                        well_name=curr_well,
                                                        pump_series=pump_series[
                                                            curr_well])

                            # If path and sheet given instead
                            elif pump_path is not None:
                                # Updating model with new pumping scenario
                                model = pastas_setparam(model, pump_path=pump_path,
                                                        pump_sheet=pump_sheet)

                        else:

                            # If pumping time series given
                            if pump_path is None:
                                if pump_series is not None:
                                    model = pastas_setparam(model,
                                                            initoptiparam=initoptiparam.
                                                            loc[
                                                                curr_well],
                                                            pump_series=pump_series[
                                                                curr_well],
                                                            well_name=curr_well,)

                                else:
                                    model = pastas_setparam(model,
                                                            initoptiparam=initoptiparam.
                                                            loc[
                                                                curr_well],
                                                            well_name=curr_well,)

                            # If path and sheet given instead
                            else:

                                # Updating model with new pumping scenario
                                model = pastas_setparam(model,
                                                        initoptiparam=initoptiparam.
                                                        loc[
                                                            curr_well],
                                                        well_name=curr_well,
                                                        pump_path=pump_path,
                                                        pump_sheet=pump_sheet)

                    # If only PD and NB models
                    if np.logical_and("_PD" in Pastasfiles[0],
                                      "_NB" in Pastasfiles[1]):

                        # Identifies missing well and index
                        missing = "NL"

                        # If PD
                        if num_model == 0:

                            proxy_name = "Proxy BK"

                        # If NB
                        else:

                            proxy_name = "Proxy NL"

                        temp = model.simulate(tmin="1950", tmax=tmax,
                                              warmup=365*30, return_warmup=False)
                        temp = temp.rename(proxy_name)
                        well_data.append(temp)

                        # Loads model, PD
                        temp = temp.rename(curr_well)
                        well_data.append(temp)

                    # If only two files
                    # If first file in sorted list is PD, next and NL
                    # missing BK and NB
                    if np.logical_and("_PD" in Pastasfiles[0],
                                      "_NL" in Pastasfiles[1]):

                        # Identifies missing well and index
                        missing = "NB"

                        # If PD
                        if num_model == 0:

                            proxy_name = "Proxy BK"
                            temp = model.simulate(tmin="1950", tmax=tmax,
                                                  warmup=365*30,
                                                  return_warmup=False)
                            temp = temp.rename(proxy_name)
                            well_data.append(temp)

                            # Loads model, PD
                            temp = temp.rename(curr_well)
                            well_data.append(temp)

                        # If NB
                        else:

                            proxy_name = "Proxy NB"
                            temp = model.simulate(tmin="1950", tmax=tmax,
                                                  warmup=365*30,
                                                  return_warmup=False)
                            temp = temp.rename(curr_well)
                            well_data.append(temp)

                            # Loads model, NL as proxy for NB
                            temp = temp.rename(proxy_name)
                            well_data.append(temp)

                # If only two files
                # If first file in sorted list is NL, next and NB
                # missing BK and PD

                    if np.logical_and("_NL" in Pastasfiles[0],
                                      "_NB" in Pastasfiles[1]):

                        # Identifies missing well and index
                        missing = "PD"

                        # If NL
                        if num_model == 0:

                            temp = model.simulate(tmin="1950", tmax=tmax,
                                                  warmup=365*30,
                                                  return_warmup=False)
                            temp = temp.rename("Proxy BK")
                            well_data.append(temp)

                            # Loads model, NL as proxy for PD, BK
                            temp = temp.rename("Proxy PD")
                            well_data.append(temp)

                            # Loads model, NL
                            temp = temp.rename(curr_well)
                            well_data.append(temp)

                        # NB
                        else:

                            temp = model.simulate(tmin="1950", tmax=tmax,
                                                  warmup=365*30,
                                                  return_warmup=False)
                            temp = temp.rename(curr_well)
                            well_data.append(temp)

            # If only 1 file
            # missing others
            if lenfiles == 1:

                missing = "others"

                # For each well model
                for num_model in range(len(well_names)):

                    # Loads model, PD as proxy for BK
                    model = models[num_model]

                    curr_well = well_names[num_model]

                    # If changing pumping scenario
                    if pumpflag == 1:

                        # If providing optimal parameters from ESMDA
                        if initoptiparam is None:

                            # If pumping time series given
                            if pump_series is not None:
                                model = pastas_setparam(model,
                                                        well_name=curr_well,
                                                        pump_series=pump_series[
                                                            curr_well])

                            # If path and sheet given instead
                            elif pump_path is not None:
                                # Updating model with new pumping scenario
                                model = pastas_setparam(model, pump_path=pump_path,
                                                        pump_sheet=pump_sheet)

                        else:

                            # If pumping time series given
                            if pump_path is None:
                                if pump_series is not None:
                                    model = pastas_setparam(model,
                                                            initoptiparam=initoptiparam.
                                                            loc[
                                                                curr_well],
                                                            pump_series=pump_series[
                                                                curr_well],
                                                            well_name=curr_well,)

                                else:
                                    model = pastas_setparam(model,
                                                            initoptiparam=initoptiparam.
                                                            loc[
                                                                curr_well],
                                                            well_name=curr_well,)

                            # If path and sheet given instead
                            else:

                                # Updating model with new pumping scenario
                                model = pastas_setparam(model,
                                                        initoptiparam=initoptiparam.
                                                        loc[
                                                            curr_well],
                                                        well_name=curr_well,
                                                        pump_path=pump_path,
                                                        pump_sheet=pump_sheet)

                    if "BK" in curr_well:
                        temp = model.simulate(tmin="1950", tmax=tmax,
                                              warmup=365*30,
                                              return_warmup=False)
                        temp = temp.rename(curr_well)
                        well_data.append(temp)

                        # Others proxy
                        temp = temp.rename("Proxy PD")
                        well_data.append(temp)

                        # Others proxy
                        temp = temp.rename("Proxy NL")
                        well_data.append(temp)

                        # Others proxy
                        temp = temp.rename("Proxy NB")
                        well_data.append(temp)

                    elif "PD" in curr_well:

                        temp = model.simulate(tmin="1950", tmax=tmax,
                                              warmup=365*30,
                                              return_warmup=False)
                        temp = temp.rename("Proxy BK")
                        well_data.append(temp)

                        # Only head value as proxy for others that are
                        # missing
                        temp = temp.rename(curr_well)
                        well_data.append(temp)

                        # Proxy
                        temp = temp.rename("Proxy NL")
                        well_data.append(temp)

                        # Proxy
                        temp = temp.rename("Proxy NB")
                        well_data.append(temp)

                    elif "NL" in curr_well:

                        temp = model.simulate(tmin="1950", tmax=tmax,
                                              warmup=365*30,
                                              return_warmup=False)
                        temp = temp.rename("Proxy BK")
                        well_data.append(temp)

                        # Proxy
                        temp = temp.rename("Proxy PD")
                        well_data.append(temp)

                        # Only head value as proxy for others that are
                        # missing
                        temp = temp.rename(curr_well)
                        well_data.append(temp)

                        # Proxy
                        temp = temp.rename("Proxy NB")
                        well_data.append(temp)

                    elif "NB" in curr_well:

                        temp = model.simulate(tmin="1950", tmax=tmax,
                                              warmup=365*30,
                                              return_warmup=False)
                        temp = temp.rename("Proxy BK")
                        well_data.append(temp)

                        # Proxy
                        temp = temp.rename("Proxy PD")
                        well_data.append(temp)

                        # Proxy
                        temp = temp.rename("Proxy NL")
                        well_data.append(temp)

                        # Only head value as proxy for others that are
                        # missing
                        temp = temp.rename(curr_well)
                        well_data.append(temp)

        # No missing wells
        else:

            missing = None

    # If using only available heads
    else:
        missing = None

        # Needs all four wells if proxyflag is not on
        if lenfiles < 4:

            sys.exit("Needs all four wells if proxyflag is not on")

    # If there is no missing data
    if isinstance(missing, type(None)):

        # For each well model
        for num_model in range(len(well_names)):

            # Loads model, PD as proxy for BK
            model = models[num_model]

            curr_well = well_names[num_model]

            # If changing pumping scenario
            if pumpflag == 1:

                # If providing optimal parameters from ESMDA
                if initoptiparam is None:

                    # If pumping time series given
                    if pump_series is not None:
                        model = pastas_setparam(model,
                                                well_name=curr_well,
                                                pump_series=pump_series[
                                                    curr_well])

                    # If path and sheet given instead
                    elif pump_path is not None:
                        # Updating model with new pumping scenario
                        model = pastas_setparam(model, pump_path=pump_path,
                                                pump_sheet=pump_sheet)

                else:

                    # If pumping time series given
                    if pump_path is None:
                        if pump_series is not None:
                            model = pastas_setparam(model,
                                                    initoptiparam=initoptiparam.
                                                    loc[
                                                        curr_well],
                                                    pump_series=pump_series[
                                                        curr_well],
                                                    well_name=curr_well,)

                        else:
                            model = pastas_setparam(model,
                                                    initoptiparam=initoptiparam.
                                                    loc[
                                                        curr_well],
                                                    well_name=curr_well,)

                    # If path and sheet given instead
                    else:

                        # Updating model with new pumping scenario
                        model = pastas_setparam(model,
                                                initoptiparam=initoptiparam.loc[
                                                    curr_well],
                                                well_name=curr_well,
                                                pump_path=pump_path,
                                                pump_sheet=pump_sheet)

            temp = model.simulate(tmin="1950", tmax=tmax,
                                  warmup=365*30, return_warmup=False)
            temp = temp.rename(curr_well)
            well_data.append(temp)

    # Well data with matching dates only
    well_data_dates = functools.reduce(lambda left, right:
                                       pd.merge(left, right,
                                                left_index=True,
                                                right_index=True),
                                       well_data)
    well_data_dates = well_data_dates[
        (well_data_dates.index.year >= int(tmin)) &
        (well_data_dates.index.year <= int(tmax))]

    # All well data
    all_well4_data = functools.reduce(lambda left, right:
                                      pd.concat([left, right], axis=1,
                                                sort=True),
                                      well_data)

    # well_data_dates - Well data with matching dates only
    # all_well4_data - all well data no matter the date
    return well_data_dates, all_well4_data


# %%###########################################################################
# Sets initial condition for subsidence run
##############################################################################

# Assuming has data for all four aquifers
# Assuming conceptual model of clay above BK, between BK and PD, PD and NL, NL
# and NB for a total of 4 clay layers.
def set_ic(headb, headt, i, mode, fullheadt, fullheadb,
           tmin, tmax, SS_data, wellnest, aq_namet, aq_nameb,
           Kv_cl, Sskv_cl, Sske_cl, Sske_aqt, Sske_aqb, CC, Nz,
           Thick_cl, nclay, Thick_aqt, Thick_aqb, Nt,
           all_well_data=None):
    """Runs groundwater models for aquifers to set initial conditions for clay
    layers

    headb - head of bottom aquifer
    headt - head of top aquifer
    i - current clay layer (1 - 4) where 1 is the top clay layer
    mode - raw groundwater data or time series from pastas (raw needs to
    to be interpolated). options: raw, pastas
    all_well_data - raw observed groundwater data not within tmin and tmax
    fullheadt - all Pastas simulated groundwater data from top aquifer despite the
    date
    fullheadb - all Pastas simulated groundwater data from bottom aquifer despite
    the date
    tmin, tmax - (str) minimum and maximum year to calculate sub
    SS_data - steady state heads relative to land surface from coastal dem 2.1. SS
    heads taken from MODFLOW model. Only used with no data from 1950 onwards but
    Pastas was used to simulate from 1950 onwards. Shouldn't be used but is an
    option
    wellnest - string of well nest name
    aq_namet - name of top aquifer
    aq_nameb - name of bottom aquifer
    Kv_cl - value of vertical hydraulic conductivity of clay layer
    Sskv_cl - value of inelastic specific storage of clay layer
    Sske_cl - value of elastic specific storage of clay layer
    Sske_aqt - value of elastic specific storage of top aquifer layer
    Sske_aqb - value of elastic specific storage of bottom aquifer layer
    CC - convergence criteria
    Nz - number of nodes in the z direction
    Thick_cl - Thickness of clay layer
    nclay - number of clay model layers
    Thick_aqt - Thickness of top aquifer
    Thick_aqb - thickness of bottom aquifer
    Nt - number of time steps

    Returns
    t_ic - time for spin up run
    h_ic - head for spin up run for clay model layers
    """

    # Create daily time series
    df = pd.DataFrame(index=pd.date_range("1950-01-01",
                                          headb.index[0],
                                          freq="d"))

    # time
    # Creating time time series [0: len of time series]
    timet_ic = np.arange(len(df.index))

    # If not the first clay layer
    if i != 1:

        # First official model aquifer head in top and bottom
        headt1 = headt.iloc[0]
        headb1 = headb.iloc[0]

        # Getting subset of dates that are before tmin to be used
        # in linear interpolation of head
        # Top
        if mode == "raw":
            subsetdate_t = all_well_data.index[np.logical_and(
                ~all_well_data.iloc[:, i-2].isna(),
                all_well_data.index < headt.index[0])]
            interpdata = all_well_data.loc[
                subsetdate_t].iloc[:, i-2]
        elif mode == "Pastas":
            subsetdate_t = fullheadt.index[np.logical_and(
                ~fullheadt.isna(),
                fullheadt.index.year < int(tmin))]
            interpdata = fullheadt.loc[subsetdate_t]

        # Getting subset of index of those dates that are before
        # tmin to be used in linear interpolation of head
        subsetindex_t = []

        for j in range(len(subsetdate_t)):
            subsetindex_t = np.append(subsetindex_t,
                                      np.flatnonzero(
                                          df.index ==
                                          subsetdate_t[j]))

        # If no earlier GW obs before model start
        if len(subsetindex_t) == 0:

            # Two values and will interpolate between
            timet2_ic = [0, timet_ic[-1]]
            headt2_ic = [SS_data.loc[wellnest, aq_namet], headt1]

        # If there are GW obs before model start, uses it for
        # llnear interpolation with SS heads
        else:

            # Converting to int
            subsetindex_t = subsetindex_t.astype(int)

            # Values and will interpolate between; time for
            # interpolation
            timet2_ic = np.insert(subsetindex_t, 0, 0)
            timet2_ic = np.append(timet2_ic, timet_ic[-1])

            # SS, head before model run, and first model head
            # Values and will interpolate between
            # Top aquifer
            headt2_ic_subset = interpdata.values
            headt2_ic = np.insert(headt2_ic_subset, 0,
                                  SS_data.loc[wellnest, aq_namet])
            headt2_ic = np.append(headt2_ic, headt1)

        # Bottom
        if mode == "raw":
            subsetdate_b = all_well_data.index[np.logical_and(
                ~all_well_data.iloc[:, i-1].isna(),
                all_well_data.index < headb.index[0])]
            interpdata = all_well_data.loc[
                subsetdate_b].iloc[:, i-1]
        elif mode == "Pastas":
            subsetdate_b = fullheadb.index[np.logical_and(
                ~fullheadb.isna(),
                fullheadb.index.year < int(tmin))]
            interpdata = fullheadb.loc[subsetdate_b]

        # Getting subset of index of those dates that are before
        # tmin to be used in linear interpolation of head
        subsetindex_b = []

        for j in range(len(subsetdate_b)):
            subsetindex_b = np.append(subsetindex_b,
                                      np.flatnonzero(
                                          df.index ==
                                          subsetdate_b[j]))

        # If no earlier GW obs before model start
        if len(subsetindex_b) == 0:
            # Two values and will interpolate between
            timeb2_ic = [0, timet_ic[-1]]
            headb2_ic = [SS_data.loc[wellnest, aq_nameb], headb1]

        # If there are GW obs before model start, uses it for
        # llnear interpolation with SS heads
        else:

            # Converting to int
            subsetindex_b = subsetindex_b.astype(int)

            # Values and will interpolate between; time for
            # interpolation
            timeb2_ic = np.insert(subsetindex_b, 0, 0)
            timeb2_ic = np.append(timeb2_ic, timet_ic[-1])

            # SS, head before model run, and first model head
            # Values and will interpolate between
            # Bottom aquifer
            headb2_ic_subset = interpdata.values
            headb2_ic = np.insert(headb2_ic_subset, 0,
                                  SS_data.loc[wellnest, aq_nameb])
            headb2_ic = np.append(headb2_ic, headb1)

        # Interpolating
        headb_ic = pd.Series(np.interp(timet_ic, timeb2_ic,
                                       headb2_ic))  # Linear
        headb_ic.set_index = df.index
        headt_ic = pd.Series(np.interp(timet_ic, timet2_ic,
                                       headt2_ic))  # Linear
        headt_ic.set_index = df.index

        # Using Pastas constant d for initial condition
        # Linearly interpolated between top and bottom
        # constant d
        spacing = np.linspace(0, Nz+1, num=Nz+2, endpoint=True)
        constant_d_ic = np.interp(spacing,
                                  [0, Nz+1],
                                  [SS_data.loc[wellnest, aq_namet],
                                   SS_data.loc[wellnest, aq_nameb]])

    # If top clay layer i == 1
    else:
        # Last spin up run is the first value in the first model
        # run
        # First official model aquifer head in top and bottom
        headb1 = headb.iloc[0]

        # Getting subset of dates that are before tmin to be used
        # in linear interpolation of head
        # Bottom
        if mode == "raw":
            subsetdate_b = all_well_data.index[np.logical_and(
                ~all_well_data.iloc[:, i-1].isna(),
                all_well_data.index < headb.index[0])]
            interpdata = all_well_data.loc[subsetdate_b].iloc[:, i]
        elif mode == "Pastas":
            subsetdate_b = fullheadb.index[np.logical_and(
                ~fullheadb.isna(),
                fullheadb.index.year < int(tmin))]
            interpdata = fullheadb.loc[subsetdate_b]

        # Getting subset of index of those dates that are before
        # tmin to be used in linear interpolation of head
        subsetindex_b = []

        for j in range(len(subsetdate_b)):
            subsetindex_b = np.append(subsetindex_b,
                                      np.flatnonzero(
                                          df.index ==
                                          subsetdate_b[j]))

        # If no earlier GW obs before model start
        if len(subsetindex_b) == 0:
            # Two values and will interpolate between
            timeb2_ic = [0, timet_ic[-1]]
            headb2_ic = [SS_data.loc[wellnest, aq_nameb], headb1]

        # If there are GW obs before model start, uses it for
        # llnear interpolation with SS heads
        else:

            # Converting to int
            subsetindex_b = subsetindex_b.astype(int)

            # Values and will interpolate between; time for
            # interpolation
            timeb2_ic = np.insert(subsetindex_b, 0, 0)
            timeb2_ic = np.append(timeb2_ic, timet_ic[-1])

            # SS, head before model run, and first model head
            # Values and will interpolate between
            # Bottom aquifer
            headb2_ic_subset = interpdata.values
            headb2_ic = np.insert(headb2_ic_subset, 0,
                                  SS_data.loc[wellnest, aq_nameb])
            headb2_ic = np.append(headb2_ic, headb1)

        # Interpolating
        headb_ic = pd.Series(np.interp(timet_ic, timeb2_ic,
                                       headb2_ic))  # Linear
        headb_ic.set_index = df.index

        headt_ic = None

        # Using Pastas constant d for initial condition
        # Linearly interpolated between top and bottom
        # constant d
        spacing = np.linspace(0, Nz+1, num=Nz+2, endpoint=True)
        constant_d_ic = np.interp(spacing,
                                  [0, Nz+1],
                                  [0, SS_data.loc[wellnest, aq_nameb]])

    # print(wellnest, " Clay " + str(i) + " Initial Condition\n")
    # Calculates sub
    # Returns interpolated t, cum sub total, interp top head, bot
    # head, cum sub inelastic, head matrix with top and bottom row
    # as top and bottom aquifer (row is node, column is time)
    t_ic, _, _, _, _, h_ic = \
        calc_deformation(timet_ic, headt_ic, headb_ic, Kv_cl,
                         Sskv_cl, Sske_cl, Sske_sandt=Sske_aqt,
                         Sske_sandb=Sske_aqb, claythick=Thick_cl,
                         nclay=nclay, sandthickt=Thick_aqt,
                         sandthickb=Thick_aqb,
                         Nz=Nz, CC=CC, Nt=Nt,
                         ic=constant_d_ic)

    # t_ic - time for spin up run
    # h_ic - head for spin up run for clay model layers
    return t_ic, h_ic


# %%###########################################################################
# Runs the bulk of code of the subsidence model for the four clay layers
##############################################################################

# Assuming has data for all four aquifers
# Assuming conceptual model of clay above BK, between BK and PD, PD and NL, NL
# and NB for a total of 4 clay layers.
def run_sub(num_clay, all_well4_data, well_data_dates, mode,
            tmin, tmax, SS_data, wellnest, K_data, Sskv_data, Sske_data, CC, Nz,
            Thick_data, ic_run, sub_total, subv_total, all_results,
            well_data=None):
    """Runs code for bulk of subsidence modeling

    num_clay - number of clay layers
    Thick_data - thickness data for each well nest and layer
    all_well4_data - all well data no matter the date
    well_data_dates - well data with only overlapping dates
    mode - raw groundwater data or time series from pastas (raw needs to
    to be interpolated). options: raw, pastas
    all_well_data - raw observed groundwater data not within tmin and tmax
    tmin, tmax - (str) minimum and maximum year to calculate sub
    K_data - vertical hydraulic conductivity data for each well nest and layer
    Sske_data - elastic specific storage data for each well nest and layer
    Sskv_data - inelastic specific storage data for each well nest and layer
    SS_data - steady state heads relative to land surface from coastal dem 2.1. SS
    heads taken from MODFLOW model. Only used with no data from 1950 onwards but
    Pastas was used to simulate from 1950 onwards. Shouldn't be used but is an
    option
    wellnest - string of well nest name
    ic_run - flag if running spin up run
    sub_total - list of lists (stores results for total subsidence)
    subv_total - list of lists (stores results for inelastic sub)
    all_results - list of lists (stores all results)

    Returns
    sub_total - list of lists (stores results for total subsidence)
    subv_total - list of lists (stores results for inelastic sub)
    all_results - list of lists (stores all results)
    """

    # Keeps track of current z (bottom of layer)
    curr_z = 0

    # For each model
    for i in range(1, num_clay+1):

        # print(wellnest, " Clay " + str(i))

        # Specifies index and clay layer names
        # VSC = very soft clay, MSC = medium stiff clay, SC = stiff
        # clay, HC = hard clay

        # If clay layer BK aquifer
        if i == 1:
            clay_name = "VSC"
            aq_namet = "BK"
            aq_nameb = "BK"

        # If clay layer between BK and PD aquifer
        elif i == 2:
            clay_name = "MSC"
            aq_namet = "BK"
            aq_nameb = "PD"

        # If clay layer between PD and NL aquifer
        elif i == 3:
            clay_name = "SC"
            aq_namet = "PD"
            aq_nameb = "NL"

        # If clay layer between NL and NB aquifer
        elif i == 4:
            clay_name = "HC"
            aq_namet = "NL"
            aq_nameb = "NB"

        # Thickness data, thickness for the clay layer, and  top and
        # bottom aquifer
        Thick_cl = Thick_data.loc[wellnest, clay_name]
        Thick_aqb = Thick_data.loc[wellnest, aq_nameb]
        Thick_aqt = Thick_data.loc[wellnest, aq_namet]

        # Time for both aquifers is the same
        # If clay layer above BK, no aquifer above it
        if i == 1:

            if mode == "Pastas":

                # BK head
                # Only bottom aquifer
                fullheadt = None
                fullheadb = all_well4_data.iloc[:, i-1]
                headb = well_data_dates.iloc[:, i-1]

            elif mode == "raw":

                # BK head
                # No top aquifer, only bottom aquifer
                headb = well_data.iloc[:, i-1]

            # No top aquifer
            headt = None

            # Thickness/Specific storage of top aquifer is 0 because
            # it doesn't exist
            Thick_aqt = 0
            Sske_aqt = 0

        # All other clay layers not first or last
        elif i != 4:

            Thick_aqb /= 2  # NB aquifer not halved.
            # Not simulating clay below it

        # If not first aquifer
        if i != 1:

            if mode == "Pastas":

                fullheadb = all_well4_data.iloc[:, i-1]
                headb = well_data_dates.iloc[:, i-1]

                fullheadt = all_well4_data.iloc[:, i-2]
                headt = well_data_dates.iloc[:, i-2]

            elif mode == "raw":

                headb = well_data.iloc[:, i-1]
                headt = well_data.iloc[:, i-2]

            Sske_aqt = Sske_data.loc[wellnest, aq_namet]

        # Creating time time series [0: len of time series]
        timet = np.arange(len(headb.index))

        # Thickness of top aquifer needs to be halved
        # For all clay layers (top will be zero even if halved)
        Thick_aqt /= 2

        # Specific storage for clays, needed for DELAY CALCULATIONS
        # Inelastic (v) and elastic (e)
        Sskv_cl = Sskv_data.loc[wellnest, clay_name]
        Sske_cl = Sske_data.loc[wellnest, clay_name]
        Sske_aqb = Sske_data.loc[wellnest, aq_nameb]

        # Kv for clays (m/day)
        # Assuming Kv = Kh
        # Using Chula value for BK clay for all clay values as starting
        Kv_cl = K_data.loc[wellnest, clay_name]

        # Number of clay layers
        nclay = 1

        # Number of time steps
        Nt = 100

        # z distribution, not used for calculation
        # Only for plotting adnd reference
        # mesh points in space
        # Current z is at the bottom of the top aquifer
        curr_z += Thick_aqt * 2

        # Z distribution from bottom of top aq to
        # bottom of clay
        dz = Thick_cl/Nz
        z = np.arange(curr_z + dz/2,
                      curr_z + Thick_cl+dz/2,
                      dz)
        # Current z updated to now bottom of clay
        z = np.insert(z, 0, curr_z)
        curr_z += Thick_cl
        z = np.append(z, curr_z)

        # If running transient simulation before model run
        # to get clay heads to where they need to be
        if ic_run:

            t_ic, h_ic = set_ic(headb, headt, i, mode, fullheadt,
                                fullheadb, tmin, tmax, SS_data, wellnest,
                                aq_namet, aq_nameb, Kv_cl, Sskv_cl, Sske_cl,
                                Sske_aqt, Sske_aqb, CC, Nz, Thick_cl, nclay,
                                Thick_aqt, Thick_aqb, Nt)

            # Calculates sub
            # Returns interpolated t, cum sub total, interp top head, bot
            # head, cum sub inelastic, head matrix with top and bottom row
            # as top and bottom aquifer (row is node, column is time)
            interp_t, sub, boundaryt, boundaryb, sub_v, h = \
                calc_deformation(timet, headt, headb, Kv_cl,
                                 Sskv_cl, Sske_cl, Sske_sandt=Sske_aqt,
                                 Sske_sandb=Sske_aqb, claythick=Thick_cl,
                                 nclay=nclay, sandthickt=Thick_aqt,
                                 sandthickb=Thick_aqb,
                                 Nz=Nz, CC=CC, Nt=Nt,
                                 ic=h_ic[:, -1])

        # If not running to get initial condition
        else:

            # Calculates sub
            # Returns interpolated t, cum sub total, interp top head, bot
            # head, cum sub inelastic, head matrix with top and bottom row
            # as top and bottom aquifer (row is node, column is time)
            interp_t, sub, boundaryt, boundaryb, sub_v, h = \
                calc_deformation(timet, headt, headb, Kv_cl,
                                 Sskv_cl, Sske_cl, Sske_sandt=Sske_aqt,
                                 Sske_sandb=Sske_aqb, claythick=Thick_cl,
                                 nclay=nclay, sandthickt=Thick_aqt,
                                 sandthickb=Thick_aqb,
                                 Nz=Nz, CC=CC, Nt=Nt)
        # Well names
        if mode == "Pastas":

            # Getting well name from Pastas model file name
            well_name = all_well4_data.columns[i-1]

        elif mode == "raw":

            well_name = well_data.columns[i-1]

        # Adds subsidence to total of all clay
        # Stores records as wellnest, well, data in list
        sub_total.append([wellnest, well_name,
                          interp_t, sub])
        subv_total.append([wellnest, well_name,
                           interp_t, sub_v])

        # If running transient simulation before model run
        # to get clay heads to where they need to be
        if ic_run:

            # Saves heads in clay nodes, z distribution
            # time original (original time series (0:len(date))), date
            # Saves initial condition head and initial condition time
            all_results.append([wellnest, well_name,
                                timet, headb.index, h, z, t_ic, h_ic])

        else:

            # Saves heads in clay nodes, z distribution
            # time original (original time series (0:len(date))), date
            all_results.append([wellnest, well_name,
                                timet, headb.index, h, z])

    return sub_total, subv_total, all_results


# %%###########################################################################
# Pastas model forecasting: simulates groundwater using different
# pumping scenarios
##############################################################################

# Future simulating Pastas with different pumping scenarios
def pastas_pump(model, pump_path, pump_sheet):
    """Pastas model that is already created.

    pump_path - path to pumping excel sheet
    pump_sheet - sheet of specific pumping scenario
    """
    # Saves optimal parameters and SD
    optiparam = model.parameters["optimal"]
    stdparam = model.parameters["stderr"]
    model.del_stressmodel("well")  # Deletes previous pumping

    # Adds new pumping
    EstTotPump = pd.read_excel(pump_path, sheet_name=pump_sheet, index_col=0,
                               parse_dates=["Date"])
    EstTotPump_ = ps.StressModel(EstTotPump.Pump, rfunc=ps.Gamma(), name="well",
                                 settings="well", up=False)
    model.add_stressmodel(EstTotPump_)

    # Assigns parameters to previous optimal parameters and SD
    model.parameters["optimal"] = optiparam
    model.parameters["stderr"] = stdparam

    # Returns model
    return model


# %%###########################################################################
# Runs Pastas and subsidence models and saves data
##############################################################################

# Assuming has data for all four aquifers
# Assuming conceptual model of clay above BK, between BK and PD, PD and NL, NL
# and NB for a total of 4 clay layers.
def bkk_subsidence(wellnestlist, mode, tmin, tmax,
                   Thick_data, K_data, Sskv_data, Sske_data, CC, Nz, ic_run,
                   proxyflag, pumpflag, model_path=None, pump_path=None,
                   pump_sheet=None, pump_series=None,
                   initoptiparam=None,):
    """Calculate sub for four clay layers and four confined aquifers.

    wellnestlist - list of wellnest to calculate subsidence for
    mode - raw groundwater data or time series from pastas (raw needs to
    to be interpolated). options: raw, pastas
    tmin, tmax - (str) minimum and maximum year to calculate sub
    CC - convergence criteria
    Nz - number of nodes in the z direction
    ic_run - True or false to generate initial condition run for clays
    proxyflag - 1 if using available heads as proxy for missing heads
    pumpflag - 1 if changing pumping scenario for Pastas
    model_path - path to python models
    pump_path - path to pumping excel sheet
    pump_sheet - sheet of specific pumping scenario

    The data sets have specific names for clays and aquifers
    Thick_data - thickness of clay and aquifers
    K_data - vertical hydraulic conductivity of clay and aquifers
    Sskv - inelastic specific storage term
    Sske - elastic specific storage term
    Returns
    all_total - list of lists: all subsidence data (total and inelastic) for
    # each clay layer
    sub_total - list of lists: sub total for all four clay layers (m)
    subv_total - list of lists: inelastic sub total for all four clay layers
    # (m)
    """
    # Preallocation
    # Head time series for each  node
    all_results = []

    # Subsidence sum for all clay layers
    sub_total = []

    # Inelastic subsidence sum for all clay layers
    subv_total = []

    # CORRECTING GW HEAD DATA TO LAND SURFACE (COASTAL DEM 2.1)
    landsurf_path = os.path.join(os.path.abspath("inputs"),
                                 "LandSurfElev_GWWellLocs.xlsx")

    # Each well nest has its own Ss and K sheet
    landsurf_data = pd.read_excel(landsurf_path,
                                  sheet_name="2.1",
                                  usecols="C:F",
                                  index_col=0)

    # If running transient simulation before model run
    # to get clay heads to where they need to be
    if ic_run:

        SS_path = os.path.join(os.path.abspath("inputs"),
                               "SS_Head_GWWellLocs.xlsx")

        # Each well nest has its own Ss and K sheet
        SS_data = pd.read_excel(SS_path,
                                sheet_name="SS_Py",
                                index_col=0)

    # For each well nest in the list
    for wellnest in wellnestlist:

        # If calculating subsidence from raw groundwater data
        if mode == "raw":

            # Preprocesses wellnest groundwater data and returns dataframe
            # with data of matching dates after interpolation
            # Returns data within tmin and tmax in well_data, and data not
            # within tmin and tmax in all_well_data
            well_data, all_well_data = bkk_wellnest_preproc(wellnest,
                                                            tmin, tmax,
                                                            proxyflag)

            # Correcting obs GW to land surface
            well_data += (landsurf_data.RASTERVALU.loc[wellnest])
            all_well_data += (landsurf_data.RASTERVALU.loc[wellnest])

            # Number clay layers
            num_clay = len(well_data.columns)

        elif mode == "Pastas":

            # Get Pastas model file names for each wellnest (Should have four
            # files for each aquifer)
            Pastasfiles = [filename for filename in os.listdir(model_path)
                           if filename.startswith(wellnest) &
                           filename.endswith(".pas")]

            # Reordering from shallowest to deepest aquifer
            # Reorder well list to shallow to deep aquifers
            # BK, PD, NL, NB
            Pastasfiles = [x for y in ["_BK", "_PD", "_NL", "_NB"]
                           for x in Pastasfiles if y in x]
            lenfiles = len(Pastasfiles)

            # Loading models for good
            models, well_names, pastas_optparam = load_Pastas_models(
                Pastasfiles, model_path)

            well_data_dates, \
                all_well4_data = load_Pastas(Pastasfiles,
                                             lenfiles,
                                             proxyflag, models,
                                             well_names,
                                             model_path,
                                             pumpflag,
                                             tmin, tmax,
                                             initoptiparam=initoptiparam,
                                             pump_path=pump_path,
                                             pump_sheet=pump_sheet,
                                             pump_series=pump_series
                                             )

            num_clay = 4

        sub_total, subv_total, all_results = run_sub(num_clay,
                                                     all_well4_data,
                                                     well_data_dates, mode,
                                                     tmin, tmax, SS_data,
                                                     wellnest,
                                                     K_data, Sskv_data,
                                                     Sske_data,
                                                     CC, Nz, Thick_data,
                                                     ic_run,
                                                     sub_total, subv_total,
                                                     all_results)

    # Returns heads in clay nodes, z dist, cum sub time series for each well,
    # cum inelastic sub time series for each well, original time step
    return all_results, sub_total, subv_total


# %%###########################################################################
# Post processes data
##############################################################################

# Need to downsample sub data into daily
def bkk_postproc(wellnestlist, sub_total, subv_total, all_results):
    """Take results of calcs, cleans it, reinterpolates to original date.

    wellnestlist - list of wellnest to calculate subsidence for
    sub_total - list of lists: wellnest, well, interp t, cum sub results (m)
    subv_total - list of lists: wellnest, well, interp t, cum sub inelastic
    results (m)
    all_results - lists of lists: wellnest, well, time original (original time
    series (0:len(date))), date, heads in clay nodes, z distribution, time in
    spin up period, heads in clay nodes in spin up (last two if run
    spin up period)

    Returns: sub_total - list of lists: reinterpolated sub_total cum sub
    results (m)
    [4] index
    subvtotal - list of lists:  reinterpolated subv_total cum sub inelastic
    results (m): [4] index
    annual_data_all - lists of lists of annual total sub for all 4 clay for
    each well nest
    avg_sub_perc - lists of lists of average subsidence percentage for each well
    from total subsidence across all time steps
    """
    # Preallocation
    # Saves annual total sub for all 4 clay for all wellnests
    annual_data_all = []

    # Saves sub percentage from each well
    avg_sub_perc = []

    # For each well nest in the list
    # num_well is the index, wellnest = name
    for num_well, wellnest in enumerate(wellnestlist):

        # Total sub for all 4 clay for one wellnest
        # Resets after one well nest completed
        cumsum_4cl = np.zeros([4, len(all_results[num_well*4][2])])

        # Assumes four wells for each wellnest
        # For each well in each wellnest
        # num_well*4+i guarantees well indexing within each well nest
        for i in range(4):

            # time original (original time series (0:len(date)))
            t_og = all_results[num_well*4+i][2]

            # Reinterpolated to original time series
            # [2] has the model time series, [3] has the cum sub results
            sub_total[num_well*4+i].append(np.interp(t_og,
                                                     sub_total[num_well*4+i][2],
                                                     sub_total[num_well*4+i][3]))

            # Adding this new interpolation to cum_sum4cl
            # Each i for each well in one well nest
            cumsum_4cl[i] += sub_total[num_well*4+i][4]

            # Reinterpolated to original time series
            # [2] has the model time series, [3] has the cum sub results
            subv_total[num_well*4+i].append(np.interp(t_og,
                                                      subv_total[num_well*4+i][2],
                                                      subv_total[num_well*4+i][3]))

        #  original date series
        date = all_results[num_well*4][3]
        # For each well nest, creating new data frame for the cum total sub sum
        df_data = {"Date": date, "CumTotSum": np.sum(cumsum_4cl, axis=0)}
        df = pd.DataFrame(df_data,
                          columns=["Date", "CumTotSum"],
                          index=date)
        df["month"] = df.index.month
        df["day"] = df.index.day

        # Resampling to each year
        annual_data = df.CumTotSum[(df["month"] == 12) &
                                   (df["day"] == 31)].to_frame()
        annual_data["year"] = annual_data.index.year

        # # IMPORTANT INFO
        # # For benchmark measurements, the first year is 0, the second year is
        # # the compaction rate over that first year.
        # # For implicit Calc, the first year has a compaction rate over that
        # # year, so need to move Implicit values down one to match benchmark
        # # measurements.
        # # Index has the right years
        # First data value is the previous year at 0 compaction
        # Adds an extra year to the end
        annual_data.loc[annual_data.index[-1] + pd.offsets.DateOffset(years=1)] = 0
        annual_data = annual_data.shift(1)  # Shifts all values down one year
        annual_data.iloc[0] = 0  # Sets first value as 0
        annual_data.index = annual_data.index.shift(-12, freq="M")

        # Adding annual rates
        annual_data["AnnRates"] = annual_data.CumTotSum.diff()

        # Saving annual data for all well nests
        annual_data_all.append([wellnest, annual_data])

        # Looking at sub percentages for each well
        all_clay = np.sum(cumsum_4cl, axis=0)
        for j in range(4):
            avg_sub_perc.append([wellnest, j,
                                 (np.average(np.divide(cumsum_4cl[j, 1:],
                                                       all_clay[1:])))])
    # Returning
    return sub_total, subv_total, annual_data_all, avg_sub_perc
