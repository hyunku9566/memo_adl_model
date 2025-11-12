Memo Model Pipeline
===================

Overview
--------
This repository builds an end-to-end pipeline that predicts human activities from CASAS-style smart-home sensor logs. The workflow is:

1. Normalize the raw CSV logs located in `data/raw/` using `preprocess.py`.
2. Generate sliding-window representations from the normalized stream.
3. Train a deep sequence model (Transformer encoder) that consumes the raw event windows, with an optional sensor skip-gram initialization step.
4. Export checkpoints + metrics for both the deep model and the lighter logistic-regression baseline.

Data Preparation
----------------
`preprocess.py` merges every CSV in `data/raw/` into a single chronological file at `data/processed/events.csv` using the schema:

| column         | description                                                        |
|----------------|--------------------------------------------------------------------|
| `timestamp`    | ISO timestamp with microsecond resolution                          |
| `sensor`       | Sensor identifier (e.g., `KitchenMotionA`)                         |
| `value_raw`    | Original value string as found in the CASAS logs                   |
| `value_type`   | One of `state`, `numeric`, `string`, or `missing`                  |
| `value_numeric`| Parsed float when `value_type == numeric`, otherwise empty         |
| `value_state`  | Upper-case categorical state (`ON`, `OFF`, `OPEN`, …) if available |
| `activity`     | Ground-truth activity label when annotated                         |
| `activity_phase`| Optional tag such as `begin`/`end`                                |
| `source_file`  | Originating raw CSV filename                                       |

Feature Engineering
-------------------
`model/features.py` consumes the processed events and builds samples as follows:

* Keep only rows that contain an `activity` label.
* For each labeled row `i`, take the `window_size` most recent events ending at `i`.
* Encode:
  - Normalized event frequencies for every sensor inside the window.
  - One-hot indicators for the current sensor, state, and value type.
  - Numeric measurement value (if present) plus a flag showing whether a numeric reading was observed.
  - Cyclical encodings of time-of-day and day-of-week.
  - Duration in minutes covered by the window.

With 35 sensors observed in the source data, the resulting feature vector has 89 floating-point values. All categorical vocabularies (sensors, states, activities, …) are stored inside the accompanying metadata so that future inference passes can recreate the exact encoding logic.

Sequence Transformer Model
--------------------------
`model/sequence_model.py` implements a Transformer encoder tailored for fixed-length event windows:

* Inputs: `(window_size × features)` tensors composed of
  - sensor IDs, state IDs, value-type IDs (learned embeddings),
  - numeric values + presence flags (projected through an MLP),
  - time-of-day & day-of-week sine/cosine encodings (projected embeddings).
* The concatenated representation is linearly projected to `model_dim`, combined with a learnable positional embedding, and passed through `num_layers` Transformer encoder blocks (`norm_first=True`, GELU activations, dropout).
* Classification uses the final timestep embedding (which corresponds to the labeled event) followed by a LayerNorm → GELU → linear head.
* Optional: initialize sensor embeddings with skip-gram weights (`--sensor-embedding-checkpoint`). The script aligns vocabularies automatically; unmatched sensors fall back to random init. You can also freeze these embeddings via `--freeze-sensor-embedding`.

Key hyperparameters (set in `train/train_sequence_model.py`):

| argument                   | default |
|---------------------------|---------|
| `--window-size`           | 50 events |
| `--batch-size`            | 512 |
| `--epochs`                | 20 |
| `--learning-rate`         | 3e-4 |
| `--model-dim`             | 128 |
| `--num-heads`             | 4 |
| `--num-layers`            | 2 |
| `--dropout`               | 0.2 |

Training command (GPU recommended):

```bash
python train/train_sequence_model.py \
  --events-csv data/processed/events.csv \
  --checkpoint checkpoint/activity_transformer.pt \
  --window-size 50 \
  --batch-size 512 \
  --epochs 20 \
  --learning-rate 3e-4 \
  --sensor-embedding-checkpoint checkpoint/sensor_embeddings.pt  # optional
```

The script handles train/val/test chronological splits, class-balanced loss weights, Transformer training, and emits accuracy/F1 metrics plus per-epoch logs to `checkpoint/activity_transformer.metrics.json`.

Baseline Logistic Model
-----------------------
`model/activity_model.py` still exposes a multinomial logistic regression (SGD) for quick baselines. It uses the engineered feature vectors described above and class balancing to cope with label skew. Primary hyperparameters:

| argument        | default |
|-----------------|---------|
| `--window-size` | 25 events |
| `--alpha`       | 1e-4 (L2 regularization strength) |
| `--max-iter`    | 60 passes over the data |
| `--tol`         | 1e-4 (early-stopping tolerance) |

Training Script
---------------
Use the CLI inside `train/train_activity_model.py` for the lightweight baseline:

```bash
# 1) Normalize the raw logs (runs once)
python preprocess.py

# 2) Train the baseline classifier
python train/train_activity_model.py \
  --events-csv data/processed/events.csv \
  --checkpoint checkpoint/activity_model.joblib \
  --window-size 25 \
  --max-iter 60 \
  --alpha 1e-4
```

The script will:

1. Load the events and build feature matrices (optionally capped via `--max-samples` for quick experiments).
2. Split samples chronologically into train/validation/test (80/10/10).
3. Fit the logistic regression classifier using the training split.
4. Evaluate on validation and test sets, printing accuracy and F1 scores plus a condensed classification report.
5. Save the fitted model, vocabularies, and feature configuration to `<checkpoint>.joblib`, alongside a JSON file containing the recorded metrics.

For the deep Transformer, refer to `train/train_sequence_model.py` as shown above. It saves:

* `checkpoint/activity_transformer.pt` – PyTorch checkpoint with model weights, vocabularies, normalization stats, and CLI args.
* `checkpoint/activity_transformer.metrics.json` – timing plus train/val/test metrics (accuracy, macro/weighted F1, per-epoch logs).

Skip-Gram Sensor Embeddings
---------------------------
Use `train/train_skipgram.py` to learn dense sensor embeddings with a skip-gram objective (negative sampling). This script targets GPU hosts (e.g., aiot-gpu) but gracefully falls back to CPU when CUDA is unavailable.

```bash
# Example GPU-friendly invocation
python train/train_skipgram.py \
  --events-csv data/processed/events.csv \
  --checkpoint checkpoint/sensor_embeddings.pt \
  --embedding-dim 128 \
  --context-size 4 \
  --negatives 5 \
  --batch-size 8192 \
  --epochs 10 \
  --learning-rate 0.01
```

What it does:

1. Loads the processed events and extracts the chronological sensor-ID stream (optionally truncated via `--max-tokens` for quick smoke tests).
2. Uses on-the-fly window sampling to create positive (center, context) pairs and draws negative samples from a 0.75-power unigram distribution.
3. Trains a two-embedding skip-gram model using Adam, automatically selecting `cuda` when available.
4. Logs per-epoch loss, throughput (pairs/second), and wall-clock timings for loading, tokenization, training, and checkpointing.
5. Saves the learned embeddings, vocabularies, hyperparameters, and speed report for downstream reuse or benchmarking.

Artifacts:

* `checkpoint/sensor_embeddings.pt` – PyTorch checkpoint containing the embedding matrix, sensor vocabulary, training config, and epoch stats.
* `checkpoint/sensor_embeddings.metrics.json` – JSON log with timing breakdowns (`load_events`, `build_tokens`, `train_skipgram`) plus per-epoch loss/throughput for “속도 계산”.
* If PyTorch cannot be imported (e.g., CPU-only dev box), the script emits a `.npz` file instead so you can still inspect the embeddings while deferring full GPU runs to aiot-gpu.

Outputs
-------
* `data/processed/events.csv` – normalized chronological event log.
* `checkpoint/activity_model.joblib` – serialized artifact with:
  - trained scikit-learn estimator,
  - vocabularies for sensors/states/value types/activities,
  - feature configuration (window size, feature width, etc.).
* `checkpoint/activity_model.metrics.json` – validation/test metrics emitted during the most recent training run (includes timing info).
* `checkpoint/activity_transformer.pt` – Transformer-based deep classifier (PyTorch state dict + metadata).
* `checkpoint/activity_transformer.metrics.json` – metrics/timers for the Transformer run.
* `checkpoint/sensor_embeddings.pt` – skip-gram embeddings + metadata (GPU-preferred).
* `checkpoint/sensor_embeddings.npz` – NumPy-based fallback when PyTorch isn’t available locally.
* `checkpoint/sensor_embeddings.metrics.json` – timer/throughput report for the skip-gram training run.

Next Steps
----------
This baseline can be extended by:

* Adding richer sequential encoders (e.g., RNNs or Transformers) once a deep-learning runtime is available.
* Incorporating longer temporal contexts or absolute time gaps as additional numerical features.
* Building an inference script that reads a live event stream, buffers the last `window_size` events, and calls the stored artifact for online activity recognition.
