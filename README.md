# Models, data, and graphical results
## Submitted *Groundwater* publication titled "Hybrid data-driven and physics-based modeling of groundwater and subsidence with an application to Bangkok, Thailand"
## Authors: Jenny T. Soonthornrangsan<sup>1</sup>, Mark Bakker<sup>2</sup>, Femke C. Vossepoel<sup>1</sup>
##### <sup>1</sup>Department of Geoscience & Engineering, Delft University of Technology, Stevinweg 1, 2628 CN Delft, The Netherlands
##### <sup>1</sup>Department of Water Management, Delft University of Technology, Stevinweg 1, 2628 CN Delft, The Netherlands
<br />
<br />
<br />

Various python scripts are provided that create different graphical results. 

- Figures.py: Produces the figures shown in the main text of the paper

- AnnSub_BarPlot_1978-2020.py: Bar graphs of annual subsidence (cm) for each well nest during 1978-2020 (Shown in the main text and supplemental information)

- AnnSub_LineSens_1978-2020.py: Line graphs of annual subsidence (cm) for sensitivity analyses of each parameter (Sskv, Sske, K, thickness) for one well nest (long run time so only calculating for one well nest at a time) (Shown in supplemental information)

- CumSub_LineForecast_1978-2060.py: Line graphs of cumulative subsidence (cm) into the future depending on the pumping scenario for each well nest during 1978-2060 (Shown in the main text and supplemental information)

- Pastas_ModelGraphs_1950-2020.py: Creates Pastas models with the option to save and import the model as well as produces graphical results shown in the paper and supplemental information. Models simulate groundwater for each well nest

- Pastas_ResultsMaps_1950-2020.py: Creates spatial maps that show the groundwater RMSE and t<sub>90</sub> results for each well in each well nest. Imports previously created Pastas models

- Sub_RMSE_Map_1978-2020.py: Creates a spatial map that show the subsidence RMSE for each well nest

- main_functions.py: Contains some functions that help with pre-processing data and creating graphs

<br />
<br />
<br />

`figures\` : graphs created from the scripts

`models\`: Pastas models created from Pastas_ModelGraphs_1950-2020.py script (Model files end with .pas)

`inputs\`: inputs needed for scripts 

- Groundwater observations for each well nest (`LC******.xlsx`)

- Subsidence observations for each well nest (SurveyingLevels.xlsx)

- Parameters required for subsidence calcultion for each well nest e.g. Sske, Sskv, K, thickness (SUBParameters.xlsx)

- Initial estimate of steady state heads for each well nest (SS_Head_GWWellLocs.xlsx)

- Land surface elevation for each well nest (LandSurfElev_GWWellLocs.xlsx)
   - Coastal DEM 2.1: https://www.climatecentral.org/coastaldem-v2.1
   - Kulp, S. A., and B. H. Strauss. 2021. CoastalDEM v2. 1: A high-accuracy and high-resolution global coastal elevation model trained on ICESat-2 satellite lidar.

- Location, well depth, screen depth, available observation years for each well for each well nest (GroundwaterWellLocs.xls)

- Basin-wide pumping data 1954-2060 (BasinPumping.xlsx)
  - 500,000 m<sup>3</sup>/day scenario: EstTotalPump_54-60_Int50 sheet
  - 250,000 m<sup>3</sup>/day scenario: EstTotalPump_54-60_IntF25 sheet
  - 1,000,000 m<sup>3</sup>/day scenario: EstTotalPump_54-60_IntF100 sheet
  - 500,000 to 250,000 m<sup>3</sup>/day in 2040 scenario: EstTotalPump_54-60_IntF50_25 sheet
  - No pumping scenario: EstTotalPump_54-60_IntF0 sheet



