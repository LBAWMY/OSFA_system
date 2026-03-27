## About this implementation

This repo's Trans-SVNet implementation is adapted from the original project: [xjgaocs/Trans-SVNet](https://github.com/xjgaocs/Trans-SVNet). You can refer to it for **training** details and the original pipeline.

Here we provide **training weights on the AutoLaparo dataset**. The AutoLaparo dataset can be requested/downloaded from [autolaparo.github.io](https://autolaparo.github.io/).

You can run a quick demo inference with `run.py`.

## Run `run.py` (single local video inference)

This script runs **frame-level phase prediction** on the video using:

- **Embedding** (ResNet50 spatial embedding)
- **TeCNO (MS-TCN)** for temporal modeling
- **Transformer head** (`Transformer2_3_1`)

It outputs a CSV with **frame id / gt / pred**.

### 1) Prepare files

- **Video**: e.g. `./data/020.mp4`
- **Label txt**: e.g. `./data/020.txt`

Label txt format must be exactly:

```text
Frame  Phase
0001   2
0002   2
0003   2
...
```

Notes:

- Labels in txt are **1-based** (Phase starts from 1). The script will convert them to **0-based** internally to match model prediction indices.

### 2) Prepare model checkpoints

By default the script expects these paths (relative to `phase_recognition/TransSVNet/`):

- `./Model/latest_model_36.pth` (embedding checkpoint)
- `./Model/latest_model_8.pth` (TeCNO checkpoint)
- `./Model/5.pth` (Transformer checkpoint)

If your checkpoints are elsewhere, pass them with arguments below.

### 3) Run

From repo root:

```bash
python run.py \
  --video_path ./data/020.mp4 \
  --label_path ./data/020.txt \
  --save_dir ./data/ \
```

### Arguments

- **`--video_path`**: path to the input video
- **`--label_path`**: path to label txt (format above)
- **`--save_dir`**: output folder


### Output

The script writes:

- `"<save_dir>/<video_stem>_Phase_Label_Pred.csv"`

CSV columns:

1. **frame_id** (1-based index into the original decoded video)
2. **gt** (0-based after conversion)
3. **pred** (0-based)
