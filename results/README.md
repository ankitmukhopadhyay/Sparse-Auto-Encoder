# Results

## Files

| File | Contents | Written by |
|---|---|---|
| `config.json` | Model, SAE release, target layers, dtype and seed. Read by notebooks 02 through 05 | notebook 01 |
| `discriminative_features.json` | Top 10 feature indices and selectivity scores per category at layer 6 | notebook 03 |
| `figures/` | Every figure in the report and README | notebooks 00, 03, 04, 05 |
| `feature_activations/` | The 900 raw feature vectors as `.npz`, one file per layer | notebook 02 |

`feature_activations/` is gitignored. Three layers of 300 x 24,576 float arrays come to about 85 MB, which does not belong in a git repository. Run notebook 02 to regenerate them; it takes roughly 37 seconds on an RTX 3050 Ti.

## config.json

```json
{
  "model_name": "gpt2",
  "n_layers": 12,
  "d_model": 768,
  "n_heads": 12,
  "target_layers": [2, 6, 10],
  "sae_release": "gpt2-small-res-jb",
  "sae_width": 24576,
  "model_device": "cuda",
  "model_dtype": "bfloat16",
  "seed": 42
}
```

This is the single source of truth for model and SAE selection. Change it in notebook 01 and the rest of the pipeline follows.

## discriminative_features.json

```json
{
  "math": {
    "indices": [20655, 3547, 14494, ...],
    "scores":  [0.9879, 0.7816, 0.7626, ...]
  },
  ...
}
```

Selectivity for feature *i* under category *c* is the mean activation of *i* over prompts in *c* minus its mean over prompts in every other category. Indices are positions in the 24,576 wide SAE dictionary at `blocks.6.hook_resid_pre`, so they resolve directly on Neuronpedia:

```
https://www.neuronpedia.org/gpt2-small/6-res-jb/<index>
```

Notebook 04 reads this file to pick the feature directions it patches.

## Figures

| File | What it shows |
|---|---|
| `category_similarity.png` | Cosine similarity between category mean activations, one 6x6 heatmap per layer |
| `sparsity_analysis.png` | L0 active feature counts per category, boxplots at layers 2, 6 and 10 |
| `separability_by_layer.png` | Fisher discriminant ratio against layer |
| `feature_heatmap_layer6.png` | Top discriminative features against category means, from notebook 03 |
| `feature_heatmap_final.png` | The same heatmap regenerated for the final report |
| `patching_by_layer.png` | KL divergence from the clean run against the layer patched, all 12 layers |
| `umap_final_report.png` | UMAP of all 300 layer 6 feature vectors, colored by category |
| `umap_layer6.html` | The interactive Plotly version of the same projection |
| `eda/` | Six exploratory figures from notebook 00: source label distributions, token lengths, and the old synthetic set against the new real one |
