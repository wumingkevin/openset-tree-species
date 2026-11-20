# Open-set tree-species recognition on FOR-species20K

Companion code, split definitions and result tables for the manuscript

> **Closed-set accuracy does not establish deployability: open-set evaluation of
> individual-tree species recognition from point clouds**

The repository reproduces the numbers, tables and figures of the manuscript and its
supplementary material from the public FOR-species20K development set plus four
public external collections.

---

## What is and is not included

**Included**

- the complete pipeline: point clouds → renderings → splits → training → detectors →
  tables and figures;
- every split definition used in the paper: S1 (three species draws), S2 (whole genus),
  S3 (five source blocks), S4 (whole family), S5 (within-genus) and the 33-class
  reference split, plus the extension and paired splits used in the supplementary material;
- aggregated result tables (`results/tables_v2/`) and per-tree predictions
  (`results/trees_v2/`), which are small and let every statistic be recomputed without a GPU.

**Not included**

- raw point clouds and rendered arrays (`views.npy`, about 7.8 GB) — download the point
  clouds from the sources below and regenerate the renderings with the script in step 2;
- trained weights (40 folds, about 5.8 GB) — retrain with the scripts in step 4.

---

## Data

| Resource | Role in the paper | Access |
|---|---|---|
| FOR-species20K development set (17,707 trees, 33 species, 25 sources) | all internal partitions | Puliti et al. (2025), Zenodo record 13255198 (concept 13255197) |
| Belgian TLS collection (66 trees) | external transfer | Zenodo 4557401 |
| St Pancras / London plane release (97 trees) | unknown-species stress test | Zenodo 5070537 |
| TreeScanPL10k, Polish districts (Belgium-quality labels) | external transfer | Zenodo 19127709 |
| Austrian LAUTx candidate set (434 trees) | rejected: duplicates with the development set | see manuscript, Section 3.6 |

The metadata files the scripts expect are `tree_metadata_dev.csv` with columns
`treeID, species, genus, dataset, data_type, tree_H, filename`, and one row per tree.

---

## Environment

Python 3.12. The exact versions used for the reported results:

```
torch==2.9.1+cu128
torchvision==0.24.1+cu128
numpy==2.5.4
pandas==3.1.0
laspy==2.7.0
matplotlib==3.11.2
pillow==13.0.0
```

A single 12 GB consumer GPU is enough for one fold at a time. Set `TORCH_HOME` if you
want the ImageNet initialisation cached outside your home directory.

---

## Pipeline

The commands below reproduce one fold end to end. Paths are relative to the repository
root; substitute your own data locations.

### 1. Point clouds → fixed-size arrays

```bash
python code/build_npz.py \
  --metadata data/metadata/tree_metadata_dev.csv \
  --root data/dev --out data/npz \
  --points 1024 --scale 50 --sampler fps --seed 0 --workers 8
```

Each tree is centred (x, y by the mean; z on the mid-height), divided by a **fixed**
50 m constant and resampled to 1,024 points. Dividing by tree height instead would
destroy absolute size, which is one of the strongest morphological cues.

### 2. Render four views per tree

```bash
python code/render_views_fix.py \
  --metadata data/metadata/tree_metadata_dev.csv \
  --root data/dev --out data/views_fix \
  --views 4 --size 192 --workers 8
```

Four azimuths 90° apart at a fixed 20° elevation, orthographic projection, one frame
per tree (half-extent = max horizontal radius, crown height / 0.90). Channels encode
height, log point density and depth. Empty pixels are zeroed in all three channels.

**Renderer note.** An earlier version of this file zeroed pixels by testing the depth
map instead of the density map, which erased roughly the lower half of every image.
`render_views_fix.py` tests `density > 0`; the archived results were produced with the
fixed version. Verify your renderings by checking that the mean ink fraction is about
8 % and that crowns are vertically centred.

### 3. Build the evaluation splits

```bash
# S1 (three species draws), S2 (whole genus), S3 (five source blocks), S4 (whole family)
python code/make_splits_all.py --metadata data/metadata/tree_metadata_dev.csv \
  --out splits --folds 5 --s1-seeds 0 1 2

# S5: unknown species inside a retained genus, source-disjoint variant
python code/make_splits_s5.py --genus Pinus Acer Picea --disjoint-sites \
  --out splits_clean --folds 3

# S5 multi-source variant used for the source stratification
python code/make_splits_s5.py --genus Quercus --keep-known 2 \
  --out splits_clean --folds 3
```

Split files are JSON with `folds[].train`, `folds[].val` (row positions into the
**filtered** `views_fix/meta.csv`) and `unknown_treeIDs`. Pre-built copies are in
`splits/`, `splits_clean/`, `splits_reference/` and `splits_paired/`.

### 4. Train one fold

```bash
python code/train_views.py \
  --views data/views_fix/views.npy --meta data/views_fix/meta.csv \
  --split splits/S1_seed0.json --fold 0 \
  --epochs 10 --batch 16 --lr 3e-4 --seed 0 \
  --out runs_v2/S1_seed0_f0
```

A fold writes `best.pt`, `logits.npy`, `feats.npy`, `meta.csv` and `classes.json`.
The recipe is ConvNeXt-Tiny (ImageNet initialisation), AdamW, cosine annealing,
16 trees (64 images) per batch, ten epochs, horizontal flip only, label smoothing 0.1.

**Alignment note.** Training rows are the `ok`-filtered subset of the metadata, whereas
`views.npy` keeps a row for every tree. The trainer maps filtered positions back to
array rows explicitly (`view_row`); the archived results were produced with that
mapping in place. If you modify `build()`, keep this mapping.

### 5. Detectors, metrics and tables

```bash
python code/osr.py \
  --logits runs_v2/S1_seed0_f0/logits.npy \
  --features runs_v2/S1_seed0_f0/feats.npy \
  --meta runs_v2/S1_seed0_f0/meta.csv \
  --classes runs_v2/S1_seed0_f0/classes.json \
  --split splits/S1_seed0.json --fold 0 \
  --scorers msp,energy,mahalanobis,knn --thr-mode quantile --keep-known 0.95 \
  --out results/all_v2/S1_seed0/f0

python code/collect_results_fix.py --root results/all_v2 --out results/tables_v2 --runs runs_v2
```

Operating thresholds are the 5 % quantile of **validation-known** scores, targeting
95 % known retention; real unknowns are never used to set a threshold. The diagnostic
"known loss @ 95 % OOD" row uses a retrospectively chosen threshold and is reported
separately.

### 6. External collections, audits, robustness checks, figures

```bash
# external transfer (Belgium, London, Polish districts)
python code/ext_evaluate.py --model runs_v2/all_species_f0_f0 \
  --views data/_ext/views_fix/views.npy --meta data/_ext/views_fix/meta.csv \
  --trees data/_ext/belgium_meta.csv --dump results/trees_v2/belgium.csv

# leakage audit against all renderable development trees
python code/find_duplicates_fix.py --views data/_ext/lautx_views_fix/views.npy \
  --meta data/_ext/lautx_meta.csv --label LAUTx

# test-side acquisition perturbations (Table 8)
python code/simulate_scanner_fix.py --model runs_v2/S1_seed0_f0 \
  --split splits/S1_seed0.json --fold 0 --limit 800 \
  --out results/tables_v2/scanner_probe_v2.txt

# temperature scaling (Table S6)
python code/refit_calibration.py --runs runs_v2 --out results/tables_v2/calibration_refit_v2.txt

# rejected-branch coarse recovery (Table 4)
python code/reject_branch_fix.py --runs runs_v2 --out results/tables_v2/reject_branch_v2.txt

# figures 1–3 (PNG at 300 dpi and vector PDF)
python code/make_figures.py
```

---

## Table and figure map

| Output | Produced by |
|---|---|
| Table 2 (S1 detector metrics), Table 3 (S2/S4), Table 6 (S3 blocks) | `code/osr.py` → `code/collect_results_fix.py` |
| Table 4 (S5 accepted / rejected branches) | `code/reject_branch_fix.py` |
| Table 5 (S1 vs S3 comparison) | `code/collect_results_fix.py` + `code/metrics.py` |
| Table 7 (external collections, plot bootstrap CIs) | `code/ext_evaluate.py` + `code/cluster_bootstrap.py` |
| Table 8 (acquisition perturbations) | `code/simulate_scanner_fix.py` |
| Table S3 (closed-set reference) | `code/osr.py` on `splits_reference/` |
| Table S4 (same-genus confusions) | `code/analyse_confusions.py` |
| Table S5 (London stress test) | `code/ext_novel_reject_fix.py` |
| Table S6 (temperature scaling) | `code/refit_calibration.py` |
| Tables S10–S11 (S5 scores and rejected-known cost) | `code/reject_branch_fix.py` |
| Table S12 (threshold sensitivity) | `analysis/threshold_sensitivity.py` |
| Tables S13–S14 (cross-genus, per-campaign) | `analysis/analyse_s5_extra.py`, `analysis/by_campaign.py` |
| Table S15 (source stratification) | `analysis/strat_genus_site.py` |
| Source-held-out re-analysis (§4 discussion) | `analysis/s3_allsource.py`, `analysis/s3_main_population.py` |
| Genus-probability aggregation | `analysis/genus_agg_verify.py`, `analysis/genus_agg_detail.py` |
| Paired congener-source experiment (auxiliary) | `analysis/build_paired_splits.py`, `analysis/build_paired_variant2.py`, `analysis/analyse_paired.py` |
| Figures 1–3 | `code/make_figures.py` |
| Dual-backbone agreement (§3.8) | `code/backbone_matrix.py` |

---

## Reproducibility notes

1. **Thresholds** are always derived from validation-known scores of the same fold.
   Unknown trees are never used to set an operating point, except where a row is
   explicitly labelled as a retrospective diagnostic.
2. **Pooled versus equal-weight summaries.** Per-detector branch recalls in the main
   tables pool tree–model records; the supplementary tables that report fold means say
   so in their captions. The two conventions differ in the second decimal and, for the
   small source-held-out groups, can differ in sign of a contrast.
3. **Bootstrap intervals** resample plots, not trees, with 2,000 replicates; plots are
   sorted and each collection gets its own seed so intervals are reproducible.
4. **Repeated predictions.** S5 has three folds over the same 587 trees; the 1,761
   tree–model records are repeated evaluations, not independent trees.
5. **Taxonomic labels.** Family labels are inferred from the binomial, not annotated
   independently in the dataset.

---

## Citation

If you use this code, please cite the manuscript. Machine-readable metadata is in
`CITATION.cff`.

## License

MIT — see `LICENSE`.

## Contact

Zaocai Huang, School of Geomatics and Geoinformation, Guilin University of Technology,
Guilin 541004, China. E-mail: wm2816768003@gmail.com
