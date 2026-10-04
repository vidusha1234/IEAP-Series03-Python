# IEAP Python Series 03 – Finding Remarkable Points in a Signal

Group assignment for the IEAP Python course. We write reusable Python functions to find zero crossings and local extrema of a signal, use them to estimate the frequency of a known signal, and then repeat the analysis on a noisy signal, cleaned with a Butterworth low-pass filter.

## Authors
- Vidusha Thebuwana
- Damien Clanet


## Division of work
| Part | Content | Author |
|------|---------|--------|
| 2 | Zero-crossing and extrema functions, docstrings, unit tests | Vidusha |
| 3 | Known signal, remarkable points, frequency estimation | Vidusha |
| 4 | Noisy signal, low-pass filtering, final analysis | Damien |

## Contents
- `IEAP_python_series03_Assignment.ipynb`: the complete notebook with code, plots and explanations.

## Requirements
- Python 3
- numpy, matplotlib, scipy
- Jupyter or VS Code to run the notebook

## How to run
1. Clone the repository.
2. Install the requirements: `pip install numpy matplotlib scipy`
3. Open the notebook and choose "Run All". Results are reproducible, because the noise uses a fixed random seed (42).

## Main results
- Clean signal: frequency of 1.0 Hz.
- Noisy signal, without filtering: the estimate is wrong (about 2.3 Hz), because noise creates extra zero crossings.
- Noisy signal, low-pass filtered at 5 Hz: frequency of 1.0 Hz.