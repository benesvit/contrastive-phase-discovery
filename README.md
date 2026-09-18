# Contrastive Phase Discovery

Companion code for the article **_[TODO: article title]_** ([TODO: journal / arXiv link]).

We learn low-dimensional embeddings of lattice configurations of a two-field
(polarization **P** + "hidden" field **A**) PbZrO3 (PZO) model with a
convolutional encoder, and recover the phase diagram by clustering in the
learned embedding space rather than in the space of hand-crafted order
parameters. Two training strategies are compared:

| Notebook | Strategy | Supervision signal |
|---|---|---|
| [siamese_network.ipynb](siamese_network.ipynb) | **Discrete contrastive** (Siamese network) | Binary same/different-cluster pairs from a reference HDBSCAN labelling |
| [metric_learning.ipynb](metric_learning.ipynb) | **Continuous metric learning** | Cosine similarity between descriptor vectors — no discrete labels |

The point of the comparison: the metric-learning route never sees discrete phase
labels, so its clusters are not inherited from the reference clustering it is
compared against. Both notebooks then apply the trained encoder to a second,
independent set of gradient-optimized configurations to check that the learned
embedding transfers.

---

## Repository layout

```
.
├── requirements.txt
├── descriptors.py             # descriptor calculation + reference-clustering check
├── siamese_network.ipynb      # contrastive learning
├── metric_learning.ipynb      # continuous metric learning
│
├── common/                    #   Shared code
│   ├── data.py                #   DataWithLabels
│   └── utils.py               #   loss curves, embedding plots, interactive phase diagram
│
├── contrastive_learning/      #   Siamese / discrete-label method
│   ├── model.py               #   SiameseEncoder, SiameseNetwork, contrastive_loss, train loop
│   ├── dataset.py             #   BalancedContrastiveDataset (balanced positive/negative pairs)
│   └── losses.py              #   (identical copy of metric_learning/losses.py — see Known issues)
│
└── metric_learning/           #   Continuous-similarity method
    ├── model.py               #   Embedder, train/test steps, train loop, get_embeddings
    ├── dataset.py             #   MetricLearningDataset (config + descriptor vector)
    └── losses.py              #   continuous_contrastive_loss, wms_loss, Q_distance
```

`data/`, `models/`, `results/`, `data_pzo_8x8_sim_anneal/` and `__pycache__/` are
listed in [.gitignore](.gitignore) — the repository holds code only.

---

## Data availability

To run the notebooks you need the monte carlo configurations which are available from Zenodo [10.5281/zenodo.19681601](https://doi.org/10.5281/zenodo.19681601):

| File | Contents |
|---|---|
| `data/monte_carlo_configs.npz` | 6 000 Monte-Carlo configurations with energies, $(\alpha, \kappa)$ values and precalculated descriptors |

Place under `data/` and the notebooks run as-is. **The descriptors and
reference labels are already stored in the bundles**, so no preprocessing step is
needed to reproduce the results.

[TODO: Zenodo DOI / where to obtain the bundles.]

### Bundle format

Every bundle is an `.npz` written by `DataWithLabels.to_npz()` and read back by
`DataWithLabels.from_npz()`, with five arrays:

| Key | Shape | Meaning |
|---|---|---|
| `Configurations` | `(N, 8, 8, 4)` | Lattice configuration. Last axis is `[P1, P2, A1, A2]` — the two polarization and two hidden-field components. |
| `Values` | `(N, 2)` | Model parameters $(\alpha, \kappa)$ of the run: the axes of the phase diagram. |
| `X` | `(N, 15)` | Descriptor vector: 5 **P** order parameters, 5 **A** order parameters, 5 **P**–**A** cross-coherences (see below). Used as the *similarity target* in metric learning. |
| `Energies` | `(N,)` | Energy per lattice site. |
| `Labels` | `(N,)` | Reference cluster id; `-1` = HDBSCAN noise. |

Channel convention: the encoders take `(N, 4, 8, 8)` (channels-first), so the
datasets `permute(2, 0, 1)` on the fly. The first convolution uses
`padding_mode='circular'` to respect the periodic boundary conditions of the
lattice.

---

## Descriptors

[descriptors.py](descriptors.py) defines the hand-crafted descriptors stored in
`X` and documents how the reference labelling was obtained. The notebooks do not
import it — it is there so the stored values can be checked and regenerated.

- `get_structure_factor(P, Q)` — complex vector structure factor $S(Q)$;
  `get_total_order(P)` — norm of the mean field.
- `get_order_parameter(P, Q)` / `get_cross_coherence(P, A, Q)` — amplitude and
  power-weighted **P**–**A** coherence, accumulated over the four sign variants
  $(\pm Q_x, \pm Q_y)$.
- `get_order_parameter_rot` / `get_cross_coherence_rot` — rotationally invariant
  versions, accumulating over the four C4 rotations of $Q$ instead.
- `get_order_parameters_rot(V)` / `get_cross_coherence_parameters_rot(P, A)` —
  the 5-feature blocks actually stored in `X`, evaluated at
  $Q \in \{(0,0), (0,\pi), (\pi,\pi), (0,\pi/2), (\pi/2,\pi/2)\}$. The non-`rot`
  variants give 7-feature blocks over the full seven-$Q$ set.

Run as a script, it recomputes the descriptors for `data/monte_carlo_configs.npz`,
asserts they match the stored `X`, then reproduces the reference clustering —
UMAP (`n_neighbors=30`, `min_dist=0.1`, cosine, `random_state=42`) followed by
HDBSCAN (`min_cluster_size=20`, `cluster_selection_epsilon=2.0`) — and prints the
adjusted Rand index against the stored `Labels`:

```bash
python descriptors.py
```

---

## Installation

Python 3.11 (the notebooks were run under 3.11.9).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

[requirements.txt](requirements.txt) pins the exact versions the article results
were produced with. `numpy` is held below 2.0 because `hdbscan` and `umap-learn`
are built against the 1.x ABI, and the pinned `torch` is a CPU build — the models
are small (~100 kB checkpoints) and train on CPU in minutes.

---

## Running the notebooks

Run either notebook top to bottom from the repository root (imports are
`common.*` / `metric_learning.*`, so the working directory matters).

Both notebooks have a `train` / `save_model` switch in the training cell: set
`train = False` to skip training and load the existing checkpoint instead.

| | `siamese_network.ipynb` | `metric_learning.ipynb` |
|---|---|---|
| Encoder | `SiameseNetwork` (shared `SiameseEncoder`) | `Embedder` |
| Embedding size | 2 | 3 |
| Loss | `contrastive_loss`, margin 1.0 | `continuous_contrastive_loss`, `mu=1.0`, `ms_mining=True` |
| Similarity target | pair label from `Labels` | `Q_distance` on `X`, rescaled by the global cosine min/max |
| Batch / epochs | 64 / 100 | 128 / 200 |
| Optimizer | AdamW, lr 3e-4, wd 1e-4 | AdamW, lr 3e-4, wd 1e-4 |
| Schedule | 5 % linear warm-up → cosine anneal | 5 % linear warm-up → cosine anneal |
| Checkpoint | `results/siamese_encoder.pt` | `models/embedder.pt` |

Checkpoints store `{'model_state_dict', 'embedding_size'}`.

Embedding-space clustering at the end of each notebook uses HDBSCAN with
`min_cluster_size=30` and a small `cluster_selection_epsilon` retuned per
embedding (0.018 / 0.18 in the Siamese notebook, 0.016 / 0.01 in the
metric-learning notebook — the second value is for the optimized set). These
epsilons are the main knob if your clusters come out over- or under-merged.

---

## Module reference

### `common/data.py`
- **`DataWithLabels`** — the central container (`configurations`, `values`, `x`,
  `energies`, `labels`). `from_npz` / `to_npz` for I/O, `filter_label(l)` for a
  single-cluster view, `__getitem__(i)` for one sample as a dict,
  `set_labels`, and `augument_randomly(ratio)` for random C4 augmentation.
- **`plot(idx)`** — side-by-side quiver plot of the **P** and **A** fields of one
  configuration, hue-coded by in-plane angle (`hsv` colormap).
- **`rotate(field, degrees)`** — C4 rotation that rotates *both* the lattice
  (`np.rot90`) and the vector components (2×2 rotation matrix). Needed because
  the model's phases are only defined up to lattice symmetry.

### `common/utils.py`
- `plot_loss_curves(train, val)` — the only helper the notebooks currently call.
- `plot_conf(configuration)` — same quiver figure as `DataWithLabels.plot`, but
  for a bare array.
- `compare_clusterings(values, labels1, labels2)` — reference vs. embedding phase
  diagrams side by side.
- `plot_2d_arrows_stratified` / `plot_3d_arrows_stratified` — Plotly views of the
  embedding vectors, stratified-sampled per cluster so small clusters stay
  visible; the legend toggles clusters.
- **`InteractivePhaseDiagram(data)`** — `.show()` renders a clickable phase
  diagram in Jupyter: clicking a point plots that configuration next to it.
  Useful for assigning physical phase names to clusters by inspection.

### `metric_learning/`
- `model.py` — `Embedder`: two circular/zero-padded conv layers (4→32→64),
  global average pool, FC 64→32→16→`embedding_size`, BatchNorm + SiLU +
  dropout 0.2, L2-normalized output. Earlier architecture variants are kept
  commented out above the active class.
- `losses.py`
  - `Q_distance(Q, c_min, c_max)` — cosine similarity of descriptor vectors,
    min-max rescaled to `[0, 1]`: the continuous "how similar are these two
    configurations" target.
  - `continuous_contrastive_loss(similarity, embeddings, ...)` — contrastive loss
    with *soft* positive/negative weights obtained by passing the similarity
    through a `sigmoid` / `lin` / `tanh` gate, optional multi-similarity mining,
    averaged over the strict upper triangle. **This is the loss used in the article.**
  - `wms_loss(...)` — weighted multi-similarity loss adapted from the
    literature, kept for comparison.
- `dataset.py` — `MetricLearningDataset` (configuration + its descriptor vector,
  with optional random C4 rotation) and `make_balanced_train_test_dataset`
  (80/20 `random_split` + `DataLoader`s).

### `contrastive_learning/`
- `model.py` — `SiameseEncoder` (same architecture as `Embedder`),
  `SiameseNetwork` (applies the shared encoder to an anchor/contrast pair),
  `contrastive_loss` (standard margin loss on the pair distance), plus the train
  loop and a batched `get_embeddings`.
- `dataset.py` — `BalancedContrastiveDataset`: for each anchor, draws a positive
  (same reference cluster) or negative (different cluster) partner with
  probability `positive_ratio`, excluding HDBSCAN noise (`-1`) from both pools.

---

## Citation

```bibtex
@article{[TODO:key],
  title   = {[TODO: title]},
  author  = {Bene\v{s}, V\'it and [TODO: co-authors]},
  journal = {[TODO]},
  year    = {[TODO]},
  doi     = {[TODO]}
}
```

## License

[TODO: choose a license — e.g. MIT for code, CC-BY for data.]
