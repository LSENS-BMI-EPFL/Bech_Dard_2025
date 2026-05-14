# **Bech & Dard Figure panels reproduction**
This repository allows the reproduction of figure panels in Bech, Dard et al. eLife (2026), starting from intermediate dataset.

Intermediate dataset can be downloaded from [zenodo](zenodo) or generated starting from NWB files using [process NWB](https://github.com/LSENS-BMI-EPFL/Bech_Dard_process_NWB).

## How to use

**1. Install the conda environment**

```bash
cd path/to/Bech_Dard_plot_figures
conda env create -f bech_dard_environment.yml
conda activate bech_dard_plot
```

**2. Clone the repo or download source code, then download the processed data folder from [Zenodo](zenodo)**

Place them so the folder structure looks like this:

```
Bech_Dard_plot_figures/   ← your folder name (user choice, adjust cd below accordingly)
├── codes/                ← source code (cloned / downloaded from this repo)
│   ├── figure_files/
│   └── utils/
├── data/                 ← downloaded from Zenodo / generated using Bech_Dard_process_NWB (must be named exactly 'data')
│   ├── figure1/
│   ├── figure1_supp/
│   ├── figure2/
│   ├── figure2_supp/
│   ├── figure3/
│   ├── figure3_supp/
│   ├── figure4/
│   └── figure4_supp/
└── figures/              ← created automatically when running the figure files
```

> **Important:** figure file scripts will fail on different architecture / naming.

## Run

Run the figure files sequentially from the repository root — this produces a `figures/` folder containing the figure panels:

```bash
cd path/to/Bech_Dard_plot_figures
python -m codes.figure_files.figure1
python -m codes.figure_files.figure1_supp1
python -m codes.figure_files.figure1_supp2
```