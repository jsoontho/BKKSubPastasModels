# Models, data, and graphical results
# Submitted *Groundwater* publication titled "Hybrid data-driven and physics-based modeling of groundwater and subsidence with an application to Bangkok, Thailand"
# Authors: Jenny T. Soonthornrangsan$^1$, Mark Bakker$^2$, Femke C. Vossepoel$^1$
## $^1$Department of Geoscience & Engineering, Delft University of Technology, Stevinweg 1, 2628 CN Delft, The Netherlands
## $^1$Department of Water Management, Delft University of Technology, Stevinweg 1, 2628 CN Delft, The Netherlands

Various python scripts are provided that create different graphical results. 

- Figures.py: Produces the figures shown in the main text of the paper

- AnnSub_BarPlot_1978-2020.py: Bar graphs of annual subsidence (cm) for each well nest during 1978-2020 (Shown in the main text and supplemental information)

- AnnSub_LineSens_1978-2020.py: Line graphs of annual subsidence (cm) for sensitivity analyses of each parameter (Sskv, Sske, K, thickness) for one well nest (long run time so only calculating for one well nest at a time) (Shown in supplemental information)

- CumSub_LineForecast_1978-2060.py: Line graphs of cumulative subsidence (cm) into the future depending on the pumping scenario for each well nest during 1978-2060 (Shown in the main text and supplemental information)

- Pastas_ModelGraphs_1950-2020.py: Creates Pastas models with the option to save and import the model as well as produces graphical results shown in the paper and supplemental information. Models simulate groundwater for each well nest

- Pastas_ResultsMaps_1950-2020.py: Creates spatial maps that show the groundwater RMSE and t_90 results for each well in each well nest. Imports previously created Pastas models

- Sub_RMSE_Map_1978-2020.py: Creates a spatial map that show the subsidence RMSE for each well nest

- main_functions.py: Contains some functions that help with pre-processing data and creating graphs



