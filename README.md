# Semantic Communication via Analog RF Computing

Simulation code for the IEEE ICC 2027 paper *"Semantic Communication via Analog RF Computing: When the Radio Computes the Meaning"* (C. K. Thomas).

A shared semantic encoder is executed at the clients by analog RF computing (passive-mixer matrix-vector multiplication). Analog computing noise at every encoder layer and the uplink channel noise are treated as one semantic noise process. A closed-form water-filling allocation spreads a client energy budget over clients, encoder layers, and the uplink, guided by per-client, per-stage semantic sensitivities.

## What the script does

`sim_semcom_arf_mnist.py` runs the full MNIST experiment:

- Shared encoder (784 → 256 → 64 → 16, tanh) with pre-activation analog noise at each layer and uplink noise on the latent.
- K = 10 clients with non-i.i.d. data (Dirichlet α = 0.12) and per-client hardware/channel quality. Half perform a fine task (10-class digit), half a coarse task (digit ≥ 5).
- Offline estimation of semantic sensitivities φ by probe-noise fitting.
- Unified water-filling allocation, compared with uniform precision and client-blind (stage-only) water-filling.
- Noise-aware co-design loop that alternates retraining and re-allocation.

If MNIST cannot be downloaded, it falls back to scikit-learn's 8×8 digits with a smaller encoder.

## Run

```bash
pip install -r requirements.txt
python sim_semcom_arf_mnist.py
```

A GPU is used automatically when available. MNIST is downloaded to `./data` on first run. `main.ipynb` runs the same script in Google Colab (it mounts Google Drive, so adjust the path in the second cell).

## Outputs

Written to the working directory and also kept in `results/`:

| File | Content |
|---|---|
| `fig2_sensitivity_profile_mnist.png` | Semantic sensitivity by stage and task group |
| `fig3_distortion_energy_tradeoff_mnist.png` | Distortion–energy tradeoff and relative gains |
| `fig5_codesign_loop_mnist.png` | Noise-aware co-design loop (shown as Fig. 4 in the paper) |
| `sweep_results_mnist.json` | Energy-budget sweep data |
| `codesign_history_mnist.json` | Co-design alternation history |

## Citation

Please cite the paper once it is available.
