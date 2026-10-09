World Ocean Atlas 2023: ambient temperature and salinity at the NESCA site
==========================================================================

The files are **not** part of this repository (about 162 MB).  They are needed
only by ``experiments/stratification/run.py``.  Fetch and verify them with::

    python scripts/fetch_data.py --woa

What this is
------------
Objectively analysed annual mean **temperature** and **practical salinity**
from the World Ocean Atlas 2023 (WOA23), ``decav`` averaging period (1955-2022),
on the global **1.00-degree** grid, from NOAA NCEI.  They give the ambient
stratification N(z) at the NESCA site, Escanaba Trough, southern Gorda Ridge.

Files
-----
.. list-table::
   :header-rows: 1

   * - File
     - Variables
     - Bytes
     - SHA-256
   * - ``woa23_decav_t00_01.nc``
     - temperature (``t_an``, ``t_mn``, ...)
     - 85168080
     - ``0bd98d3c9c20254e8fcf714aefdd23b4a191eacdfdc3ce5d019205e4a8bc41b5``
   * - ``woa23_decav_s00_01.nc``
     - salinity (``s_an``, ``s_mn``, ...)
     - 77439270
     - ``3ab18000cea789a2b349cf6b08531f1ebf144f601d90a83f4c3508456003f019``

Source URLs:

- https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/temperature/netcdf/decav/1.00/woa23_decav_t00_01.nc
- https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/salinity/netcdf/decav/1.00/woa23_decav_s00_01.nc

Downloaded 2026-09-16.  Server ``Last-Modified``: 2024-01-29 15:23:47 GMT
(temperature) and 2024-01-29 14:59:38 GMT (salinity).  Both files are
netCDF-4/HDF5.

Product details (from the file headers)
---------------------------------------
- ``t00`` / ``s00``: annual climatology, one time step.
- Grid: 180 x 360 cells, centres at half degrees; longitudes -179.5 to 179.5,
  latitudes -89.5 to 89.5.
- Depth: 102 standard levels, 0-5500 m, positive down; 5 m spacing over
  0-100 m, 25 m over 100-500 m, 50 m over 500-2000 m, 100 m over 2000-5500 m.
- ``t_an`` (degrees Celsius) and ``s_an`` (practical salinity) are the
  objectively analysed means; ``_FillValue = 9.96921e+36``.  Also present per
  variable: ``_mn``, ``_dd``, ``_sd``, ``_se``.
- CF-1.6, EPSG:4326, ``date_created = 2024-01-28``.

Site
----
The NESCA push-core field spans latitude 40.6957-41.1632 N, longitude
127.5357-127.4076 W, water depth 3201-3304 m, from the USGS Escanaba Trough
core-location data release:
https://cmgds.marine.usgs.gov/catalog/pcmsc/DataReleases/ScienceBase/DR_P13B46QX/TN403_CoreLocations.html

On the 1-degree grid the vent field (about 40.98 N, 127.49 W) lies in the cell
centred at 40.5 N, 127.5 W, and the northernmost cores in the cell centred at
41.5 N, 127.5 W.  ``experiments/stratification/run.py`` reads the cells covering
the core field with a halo and reports the spread across them.

Citation
--------
The files state: "These data are openly available to the public.  Please
acknowledge the use of these data with the text given in the acknowledgment
attribute."  As US federal government data, WOA23 is in the public domain;
citation is requested.

- Reagan, J. R., Boyer, T. P., Garcia, H. E., Locarnini, R. A., Baranova,
  O. K., Bouchard, C., Cross, S. L., Mishonov, A. V., Paver, C. R., Seidov, D.,
  Wang, Z. & Dukhovskoy, D. (2024).  World Ocean Atlas 2023.  NOAA National
  Centers for Environmental Information (NCEI Accession 0270533).
- Locarnini, R. A. et al. (2023).  *World Ocean Atlas 2023, Volume 1:
  Temperature.*  NOAA Atlas NESDIS 89.  doi:10.25923/54bh-1613
- Reagan, J. R. et al. (2023).  *World Ocean Atlas 2023, Volume 2: Salinity.*
  NOAA Atlas NESDIS 90.  doi:10.25923/70qt-9574
