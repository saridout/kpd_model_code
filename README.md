# Code for "A mathematical model for ketosis-prone diabetes suggests the existence of multiple  pancreatic β-cell inactivation mechanisms"

The file "environment.yml" specifies the packages used to perform all simulations, and can be used with anaconda to create an environment matching ours.

With such an environment active, running the jupyter notebook "code/all_simulations.ipynb" will run all the simulations and save the data to the directory "data".

From this data, the notebook "code/all_plots.ipynb" can be used to generate all plots. 

To set up the environment and install it as a kernel for jupyter notebooks, run the commands:

```
conda env create --name kpd --file environment.yml
conda activate kpd
python -m ipykernel install --user --name kpd --display-name "Python (KPD)"
```

Then, start a new jupyter session and the kernel "Python (KPD)" should be selectable in the list of kernels. Alternatively, the kernel can be selected from the command line by running

```
jupyter notebook --MultiKernelManager.default_kernel_name=kpd
```

Depending on your computer's LaTeX environment, you may get LaTeX errors when plotting. You can fix such errors without downloading the needed LaTeX packages by commenting out all of the axis label and text commands..

To remove the environment, run:
```
jupyter kernelspec uninstall kpd
conda remove -n kpd --all
```


