# TVB-LB_complexity

This repository hosts the code to reproduce the results of the article "Investigating the Impact of Local Manipulations on Spontaneous and Evoked Brain Complexity Indices: A Large-Scale Computational Model", [Applied Sciences. 2024; 14(2):890](https://www.mdpi.com/2076-3417/14/2/890), by Gaglioti, G.; Nieus, T.R.; Massimini, M.; Sarasso. Along with the code to run the simulations, the output data is provided to quickly generate the figures and, if needed, perform further analyses.

## Dependencies

- The Virtual Brain (TVB), Version: 2.7.2
- Jupyter Notebook
- numpy
- matplotlib
- scipy
- statsmodels
- scikit_posthocs
- pandas
- seaborn
- networkx
  
## Table of Contents

- [Functions](#functions)
- [Data](#data)
- [Notebooks](#notebooks)

## Functions

- **func_TVB.py**: this file contains functions to run simulations of the Larter & Breakspear (LB) model in The Virtual Brain (TVB).
  
- **func_FR.py**: this file defines functions to compute the firing rate from the output of the LB model generated in TVB.
  
- **func_complexity.py**: this file contains functions to compute the complexity metrics used in the article.

## Data

This folder contains the following subfolders:

- **connectivity**: This folder contains two connectomes consisting of 76 nodes (Dconn) and 998 nodes (Hconn) used in the article and ready to be loaded into TVB. Dconn and Hconn refer to two connectomes present in the TVB dataset. They are located in the 'connectivity' folder of TVB under the names connectivity_76.zip and connectivity_998.zip, respectively. For more information, refer to [TVB Complete Dataset Description](https://docs.thevirtualbrain.org/manuals/UserGuide/Complete_Dataset_Description.html.).
  
- **outputs**: This folder contains the model outputs used for the main results and figures of the article. They can also be obtained through the code in the Jupyter Notebook **run_simulations.ipynb** (see below). The simulation of all conditions (i.e., 12 stimulation sites x 21 conditions; + Hconn simulations; + spontaneous activity), is memory and computationally intensive. You can download the complete output [here](https://unimi2013-my.sharepoint.com/personal/gianluca_gaglioti_unimi_it/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fgianluca%5Fgaglioti%5Funimi%5Fit%2FDocuments%2Foutputs&ga=1). After downloading, place it in the 'path_outputs' directory ('.../data/outputs').
  
- **stim_manip**: This folder provides two Python dictionaries containing information about the explored combinations of stimulated site and local manipulations. They are produced in the Jupyter Notebook **stimulation_&_manipulation_protocol.ipynb** (see below).

## Notebooks

- **stimulation_&_manipulation_protocol.ipynb**: This notebook demonstrates how the various stimulation and local manipulation conditions presented in the article are generated.
  
- **run_simulations.ipynb**: This notebook allows simulation of the LB model in TVB as desribed in the article. To reproduce all conditions, simply modify the stimulated site and local manipulation through the dictionaries provided in the **stim_manip** folder (see above).
  
- **exploring_outputs.ipynb**: This notebook provides a brief exploration of the model output (spontaneous and evoked activity) and how it is converted into instantaneous firing rate (IFR).
  
- **figures.ipynb**: In this notebook, the main analyses and reproductions of the main (and some supplementary) figures present in the results are performed. The data provided in this repository are sufficient to run the notebook without needing to execute all simulations.
