# GBSPE
This repository contains python code for estimating the percentage of GBS advantage in Gaussian expectation problems using improved GBS-I, improved GBS-P, and MC methods. 

## Features

- Generate Gaussian expectation problems.
- Estimate the percentage of GBS advantage by random sampling in the problem space.
- Create figures in the companion paper.

## Installation

You can install this software directly from GitHub:

```bash
pip install git+https://github.com/sshanshans/GBEPE.git
```

## Usage

The following steps recreate figures and results for small cases ($N = 3, 4$ and $K = 2, 3$).
You can increase $N$ and $K$, but be mindful of computational costs, which may take days. 

### 1. Prepare executable scripts
- `cd script`
- `chmod +x make_dict.sh run_percentage.sh make_figures.sh`

---

### 2. Precompute hafnian loop up table
Precompute the hafnian values since they are used repeatedly in the upcoming sampling process.
- `./make_dict.sh`

---

### 3. Estimate the percentage of GBS advantage
Randomly sample Gaussian expectation problems and record the percentage of GBS advantage.
- `./run_percentage.sh`

---

### 4. Make figures
Create heatmaps, convergence plots, and histograms from the companion paper.
- `./make_figures.sh`

The figures are saved in `exp/haf/fig`, `exp/haf/fig2`, `exp/hafsq/fig`, `exp/hafsq/fig2`

---

## License
This software is licensed under the GPL-3.0 License. See the LICENSE file for details.

## Citations
If you use this software in your research, please cite the associated paper:

**Estimating the Percentage of GBS Advantage in Gaussian Expectation Problems**  
Jørgen Ellegaard Andersen, Shan Shan (2025)