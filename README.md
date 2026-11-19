# CAPRI: capability-preserving federated unlearning for intrusion detection

Code, logged experiment outputs and paper source for

> I. Ghauri and O. Yamin, *Sole-Holder Deletion in Federated Intrusion Detection: Measuring, Predicting and
> Repairing Lost Attack-Family Detection*. Department of Data Science, FAST-NUCES, Islamabad, Pakistan.

When a client leaves a federated intrusion detection system and is the only holder of an attack family, federated
unlearning — including exact retraining — removes the detector's ability to recognise that family. This repository
contains:

- **CAPRI-Audit**: a pre-deletion estimate of each attack family's post-deletion recall from information the server
  already holds (label counts, stored update history, embedding separability).
- **CAPRI-Guard**: a plug-in repair that restores per-family recall from a small, independently held exemplar set,
  judged against a *lawful oracle*.
- The full benchmark: six ownership scenarios (S0–S3), seven baselines plus exemplar fine-tuning, two datasets
  (Edge-IIoTset, CICIoT2023), ablations, sensitivity analyses and extension experiments.

Every number, table and figure in the paper is computed from the logged outputs in `results/` by the scripts in this
repository; none is entered by hand.

## Repository layout

| Path | Content |
|---|---|
| `code/capri/` | Implementation (Python package) |
| `code/*.py` | Analysis and figure scripts |
| `kaggle/` | Scripts that run the experiments as GPU jobs on Kaggle |
| `results/full/`, `results/full_cic/` | Logged outputs of the 40 + 40 main runs (Edge-IIoTset, CICIoT2023) |
| `results/p3/` | Logged outputs of the 35 sensitivity and extension runs (Edge-IIoTset) |
| `results/analysis*/` | Tables (CSV), statistics (JSON) and figures computed from the run outputs |
| `results/*.json, *.md, *.csv` | Dataset log with SHA-256 hashes, target-family selection, duplicate counts, DNS-field check |
| `paper/` | LaTeX source (Springer Nature template), figures, the compiled PDF and the build scripts |

### Implementation modules (`code/capri/`)

| Module | Content |
|---|---|
| `data.py` | Loading with hash verification, cleaning, duplicate grouping, splits, federated preprocessing, automated leakage checks, CICIoT2023 sampling, target-family selection |
| `partition.py` | Ownership scenarios S0–S3 and per-client label counts |
| `model.py` | MLP detector (default) and 1D-CNN detector |
| `fl.py` | Vectorised FedAvg with restricted softmax (FedRS), early stopping, per-round update history, partial participation |
| `unlearn.py` | B1 retrain oracle, B2 lawful oracle, B3 continued training, B4 gradient ascent, B5 FedEraser, B8 exemplar fine-tuning |
| `capri.py` | CAPRI-Audit (leave-one-out estimate, signals, audit model, labels, counts rule) and CAPRI-Guard (Guard-S, Guard-F) |
| `evaluate.py`, `metrics.py` | Per-family recall, detection debt, collateral loss, silent misses, false alarms, membership inference, relearning |
| `experiment.py` | Pass 1: pre-deletion training, audit signals, baselines B0–B5 |
| `experiment2.py` | Pass 2: audit fitting, B6 distillation, B8, CAPRI-Guard, ablations, sweeps |
| `experiment3.py` | Pass 3: design sensitivity and extensions (partial participation, sequential deletion) |
| `analysis.py`, `figures.py` | Statistics and figures |

## Requirements

Python 3.11 or later:

```
pip install -r requirements.txt
```

The experiments were run on a Kaggle GPU (NVIDIA Tesla T4, Python 3.13). The analysis and the paper need no GPU.
Building the paper needs a LaTeX distribution with `pdflatex` and `bibtex` (for example MiKTeX or TeX Live).

## Data

The datasets are not redistributed; obtain them from their official sources:

- **Edge-IIoTset** (Ferrag et al., 2022): `DNN-EdgeIIoT-dataset.csv`, placed in `data/raw/edgeiiotset/`.
- **CICIoT2023** (Neto et al., 2023): `MERGED_CSV.zip` from the Canadian Institute for Cybersecurity, extracted to
  `data/raw/ciciot2023/extracted/MERGED_CSV/`.

File sizes and SHA-256 hashes are listed in `results/dataset_log.md`; the loaders verify them. The CICIoT2023 sample
(at most 20,000 unique records per class, seed 0) is built with `capri.data.build_ciciot2023_sample`.

## Reproducing the experiments

Every run is deterministic given its scenario (`S0`, `S1`, `S2a`, `S2b`, `S2c`, `S3`) and seed.

1. **Target families**: `python code/target_gate.py ciciot2023 <sample.csv> <sample.json> results/target_selection_ciciot2023.json`
   (Edge-IIoTset: `python code/target_gate.py edgeiiotset <csv> - <out.json>`).
2. **Runs**: each pass is a Kaggle GPU job built from `kaggle/kernel_template.py`:
   `python kaggle/launch.py <job-name> <pass1|pass2|pass3> '<runs json>' ['<extra json>'] [--sources <earlier jobs>]`,
   where `<runs json>` is a list of `[scenario, seed]` pairs (pass 3: `[variant, seed]`) and `'{"dataset": "ciciot2023"}'`
   selects the second dataset. Change the account and dataset names in `kaggle/launch.py` to your own. The reported runs:
   - passes 1 and 2, each dataset: S1 and S3 with seeds 0–9; S0, S2a, S2b, S2c with seeds 0–4 (40 runs);
   - pass 3, Edge-IIoTset, S3: variants `alpha0.1`, `alpha1.0`, `K20`, `cnn`, `unweighted`, `partial0.5`, `sequential`,
     seeds 0–4 (35 runs).

   The same functions run without Kaggle: `capri.experiment.run_pass1`, `capri.experiment2.run_all`,
   `capri.experiment3.run_variant`.
3. **Leakage checks** run automatically at the start of every run; a run stops if any check fails.

## Reproducing the results and the paper from the logged outputs

No GPU and no dataset are needed; everything is computed from `results/`.

```
python code/make_results.py results/full results/analysis
python code/make_results.py results/full_cic results/analysis_cic
python code/extensions_local.py results/full results/analysis
python code/extensions_local.py results/full_cic results/analysis_cic
python code/combined_results.py results/full results/full_cic results/analysis results/analysis_cic results/p3 results/analysis_combined
python code/sensitivity_figure.py results/analysis results/analysis_cic results/analysis_combined

cd paper
python tools/make_figures.py .. figures
python tools/paper_figures.py .. figures
python tools/build.py
```

`paper/tools/render.py` fills every placeholder in `paper/main_template.tex` from the analysis outputs and stops with
an error if any value cannot be computed; `build.py` then compiles `main.tex` to `main.pdf`.

## Licence

MIT; see `LICENSE`. The datasets are subject to their providers' terms.
