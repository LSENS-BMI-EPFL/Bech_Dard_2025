# **Bech & Dard Figure panels reproduction**
This repository allows the reproduction of figure panels in Bech, Dard et al. (2026), starting from intermediate dataset.

Intermediate dataset can be downloaded from [zenodo](zenodo) or generated starting from NWB files using [process NWB](https://github.com/LSENS-BMI-EPFL/Bech_Dard_process_NWB).

### Step 1: Install the conda environment

```bash
   conda create -n bech_dard_fig_env python=3.11
   conda activate bech_dard_fig_env
   pip install -r path/to/bech_dard_requirements.txt
```
    
### Step 2: Clone the repo or download source code

### Step 3: Download the processed data folder 

### Step 4: Placed the processed data folder into the 'Bech_Dard_plot_figures' folder

### Step 5: Run the figure files sequentially , it produces a 'figures' folder containing panels from the figures

```bash
   cd C:\path\to\Bech_Dard_plot_figures
   python -m codes.figure_files.figure1
   python -m codes.figure_files.figure1_supp1
   python -m codes.figure_files.figure1_supp2
```
