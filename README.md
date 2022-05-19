# The new programming complex for the vegetation scheme of COSMO-CLM model and its analysis and visualization.

### Authors:
<p align="justify">
E. Churiulin<sup>1</sup>, V. Kopeikin<sup>2</sup>, M. Übel<sup>3</sup>, J. Helmert<sup>3</sup>, J.M. Bettems<sup>4</sup>, M.H. Tölle<sup>1</sup>

1. Center for Environmental Systems Research, University of Kassel, 34117 Kassel, Germany
2. Hydrometcenter of Russia, 123242 Moscow, Russia
3. German Weather Service, 63067 Offenbach am Main, Germany
4. Federal Office of Meteorology and Climatology, Zurich, CH-8058, Switzerland

<em><strong>Correspondence to: E. Churiulin (evgenychur@uni-kassel.de)</strong></em>

## The repository description:
<p align="justify">
  The repository has the personal programing modules which were created for the improvements of the simplified vegetation scheme of the regional climate model <a href="https://wiki.coast.hzg.de/clmcom ">COSMO-CLM</a>. The new updates have the modern algorithm based on the physically Ball-Berry approach coupled with photosynthesis processes based on Farquhar and Collatz models for C<sub>3</sub> and C<sub>4</sub> plants and the "two-big leaf" approach for the photosyntetic active radiation. Moreover, there are scripts, in repository, designed to postprocess of COSMO-CLM data, process of satellite and observational data, verify and visualise experimental results. 
</p>

## The repository contains:
1. New module with stomatal resistance and leaf photosyntesis for CCLM<sub>v3.5</sub>
    * [src_phenology.f90][phen]
2. Module with constant PFT parameters and other constants for CCLM<sub>v3.5</sub>
    * [src_data_phenology.f90][data]
3. New ***Python*** project, with statistical and visualization modules:
    * [CESR_project.py][cesr] and [stat_module][main] - the main programms for verification, statistical analysis and visualization of COSMO-CLM results:  
        + [cosmo_data.py][cosmo] - personal module for downloading and preparing COSMO data
        + [fluxnet_data.py][flux] - personal module for downloading and preparing FLUXNET and EURONET data
        + [insitu_data.py][insitu] - personal module for downloading and preparing data from Linden and Lindenberg
        + [reanalysis_data][rean] - personal module for downloading and preparing reanalysis data from E-OBS, HYRAS and GLEAM datasets
        + [system_operation][sys] - personal module with a system functions for cleaning data
        + [vis_module][vis] - personal module for data visualization
        + [stat_functions][stat] - personal module for work with statistical analysis
    * [cartopy_map][cart] - personal script for figure 1a

## Financial support:
This research was funded by the German Research Foundation (DFG) through grant number 401857120  
  
  
[cesr]: https://github.com/EvgenyChur/PT-VAINT/blob/main/CESR_project.py  
[main]: https://github.com/EvgenyChur/PT-VAINT/blob/main/stat_module.py  
[vis]: https://github.com/EvgenyChur/PT-VAINT/blob/main/vis_module.py
[sys]: https://github.com/EvgenyChur/PT-VAINT/blob/main/system_operation.py
[stat]: https://github.com/EvgenyChur/PT-VAINT/blob/main/stat_functions.py
[flux]: https://github.com/EvgenyChur/PT-VAINT/blob/main/fluxnet_data.py
[insitu]: https://github.com/EvgenyChur/PT-VAINT/blob/main/insitu_data.py
[rean]: https://github.com/EvgenyChur/PT-VAINT/blob/main/reanalysis_data.py  
[cosmo]: https://github.com/EvgenyChur/PT-VAINT/blob/main/cosmo_data.py
[main_ini]: https://github.com/EvgenyChur/PT-VAINT/blob/main/main_ini.sh
[bonus]: https://github.com/EvgenyChur/PT-VAINT/blob/main/bonus_ini.sh
[E_dom]: https://github.com/EvgenyChur/PT-VAINT/blob/main/EOBS_domain.sh
[H_dom]: https://github.com/EvgenyChur/PT-VAINT/blob/main/HYRAS_domain.sh
[GL_dom]: https://github.com/EvgenyChur/PT-VAINT/blob/main/GLEAM_domain.sh  
[E_py]: https://github.com/EvgenyChur/PT-VAINT/blob/main/EOBS_python.sh
[H_py]: https://github.com/EvgenyChur/PT-VAINT/blob/main/HYRAS_python.sh
[GL_pyt]: https://github.com/EvgenyChur/PT-VAINT/blob/main/GLEAM_python.sh
[data]: https://github.com/EvgenyChur/PT-VAINT/blob/main/data_phenology.f90
[phen]: https://github.com/EvgenyChur/PT-VAINT/blob/main/src_phenology.f90
[cart]: https://github.com/EvgenyChur/PT-VAINT/blob/main/cartopy_map.py
