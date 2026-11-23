# NPS similarity and cross-kernel generalisation of CT denoisers

Code and results for the article:

> K. Halloum and H. Z., "Noise-texture similarity predicts the cross-kernel generalisation loss of deep-learning denoisers in low-dose CT: a pre-registered controlled study" (submitted).

The study tests whether the 1-Wasserstein distance (W1) between the normalised radial noise power spectra (NPS) of training and test data predicts the generalisation loss of CNN denoisers (REDCNN, DnCNN) when the reconstruction kernel changes. A continuous kernel family between D45 and B30 is emulated from the Mayo low-dose CT data; 80 models (2 architectures x 8 training conditions x 5 patient folds) are tested on 21 conditions.

## Main results

| Hypothesis | REDCNN | DnCNN | Outcome |
|---|---|---|---|
| V0: emulated vs real B30 (RMSE ratio, 90% CI) | 0.986 [0.978, 0.994] | 0.987 [0.979, 0.994] | equivalent |
| H1: W1 ranks loss (median per-patient Spearman rho) | 0.79, p_Holm = 0.004 | 0.87, p_Holm = 0.004 | confirmed |
| H2: W1 coefficient, mixed model (95% CI) | 0.772 [0.524, 1.030] (joint) | | confirmed |
| H3: direction of the larger loss | sharp -> smooth | smooth -> sharp | architecture-dependent |
| H4: prediction on off-family kernels | 23.6% vs naive 7.9% | 23.1% vs naive 7.6% | not met |
| H5: reference-free W1 (CCC) | 0.70 [0.62, 0.79] | | not met |

Full numbers: `results/stage2d_results.json`.

## Data

Mayo Clinic / AAPM 2016 Low Dose CT Grand Challenge data, as distributed on Kaggle:
https://www.kaggle.com/datasets/andrewmvd/ct-low-dose-reconstruction (original DICOM, 1 mm, D45 and B30, full and quarter dose).
**No image data are included in this repository.** Please cite McCollough et al., Med. Phys. 44(10):e339-e352, 2017, doi:10.1002/mp.12345, and respect the dataset's terms of use.

## Repository layout

```
code/
  ctphys.py              NPS (ICRU 87 / TG-233 normalisation), radial profiles, W1 distance, transfer-function estimation
  stage1_main.py         Stage 1: data audit, NPS, D45 -> B30 kernel-emulation test
  stage1b_main.py        Stage 1b: emulation robustness (leave-one-patient-out)
  stage2a_main.py        Stage 2a (CPU): hybrid kernel operator T_h, family checks, W1/dlogvar matrices, H5, training cache
  stage2b_main.py        Stage 2b (GPU): smoke tests and pilot (fixes N = 20 000 iterations)
  stage2c_lib.py         Stage 2c: models (REDCNN, DnCNN), sampler, training of the 80 models (resumable, 2 GPUs)
  stage2c_driver.py      notebook cell that runs stage2c_lib.py
  stage2d_eval.py        Stage 2d: frozen-model inference on the 21 test conditions (RMSE, SSIM, residual NPS)
  stage2d_stats.py       pre-registered statistics (V0, H1-H4), pure NumPy/SciPy
  stage2d_post.py        assembly of results, tables and figures
  stage2d_driver.py      notebook cell that runs Stage 2d end to end
  test_*.py              offline tests on synthetic data with known answers
figures/
  make_figs.py, h1_data.py   reproduce Fig. 2 and Fig. 3 of the article (PNG 300 dpi + PDF)
results/
  stage2d_results.json       all pre-registered outcomes
  metrics.csv                all 3 360 evaluations (model x test patient x test condition)
protocol/
  README.md                  pre-registration (see below)
```

## How to reproduce

The pipeline was run as private Kaggle Notebooks (GPU: 2 x NVIDIA T4). Each stage is one notebook whose input is the Kaggle dataset plus the output of the previous stage:

1. `stage1_main.py`, `stage1b_main.py` (CPU) - optional checks of the emulation.
2. `stage2a_main.py` (CPU) - writes `stage2a/` (T_h, W1 pair matrix, H5) and the training cache (`cache/`).
3. `stage2b_main.py` (GPU) - pilot; fixes the number of iterations.
4. `stage2c_driver.py` with `LIB = open("stage2c_lib.py").read()` (GPU, ~30 h in total, resumable across sessions) - writes the 80 models to `stage2c/`.
5. `stage2d_driver.py` with `LIB`, `EVAL`, `STATS`, `POST` set to the contents of `stage2c_lib.py`, `stage2d_eval.py`, `stage2d_stats.py`, `stage2d_post.py` (GPU, ~6 h) - writes `stage2d/` (metrics, statistics, figures).

Offline tests (CPU, no data needed):

```
cd code
python test_ctphys.py
python test_stage2d_stats.py
python test_stage2a.py
python test_stage1.py
python test_stage1b.py
```

Figures 2 and 3:

```
cd figures
python make_figs.py
```

## Requirements

Python 3.10+, see `requirements.txt` (NumPy, SciPy, pandas, PyTorch, pydicom, scikit-image, statsmodels, matplotlib). Versions used were those of the Kaggle Python image of November 2026.

## Licence

Code: MIT (see `LICENSE`). Figures and results: CC BY 4.0.

## Citation

See `CITATION.cff`. Please cite the article once published.
