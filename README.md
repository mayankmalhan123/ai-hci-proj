# AI for HCI Seminar - Final Project

## Description
This repository contains the description and instructions for the final project 
of the AI for HCI Seminar at Saarland University (2024). After loading the 
submodule it also contains a notebook for getting started with multi-objective 
optimization.

## Installation
Start by downloading the required submodules. Then use your terminal to create 
a new environment with the project dependencies and start the notebook. This 
environment.yml file also includes the dependencies for the getting started
materials. The environment.yml was exported and tested on macOS - if you run 
into environment creation issues try to install the packages manually. If you
still run into issues that you cannot solve by yourself reach out to us.
> cd this_directory  
> git pull --recurse-submodules  
> conda env create -f environment.yml  
> conda activate opt  
> jupyter lab


To install all the packages manually (e.g., for Windows):
> cd this_directory  
> conda create -n opt  
> conda install -c conda-forge jupyterlab  
> pip install numpy trimesh ipyvolume  
> pip install pygame imageio[ffmpeg] wonderwords  
> pip install ipython_genutils
