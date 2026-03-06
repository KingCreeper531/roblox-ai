# Roblox AI — Behavioural Cloning Pipeline

An end-to-end system that watches you play Roblox, learns your behaviour from recordings and YouTube videos, then plays autonomously.

---

## Quick Start

```bash
# Install deps and launch the web dashboard
python start.py --install
```

Then open **http://localhost:5000** in your browser.

---

## Architecture

```
Videos / YouTube          Your Gameplay
      |                        |
      v                        v
extract_frames.py       record_gameplay.py
      |                        |
      v                        v
 data/frames/           data/recordings/
  (unlabelled)           (labelled: frame + inputs)
      |                        |
      +----------+-------------+
                 |
                 v
     Phase 1: Pretrain (InverseDynamics)
     Phase 2: Finetune (BehaviouralCloning)
                 |
                 v
          models/best_model.pt
                 |
                 v
          inference.py (AI plays live)
```

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt

# ffmpeg required for yt-dlp:
# Windows: winget install ffmpeg
```

---

## Step-by-Step Usage

### Step 1 — Extract frames from videos

```bash
python extract_frames.py --url "https://www.youtube.com/watch?v=XXXX"
python extract_frames.py --source data/videos/gameplay.mp4
python extract_frames.py --urls-file my_urls.txt
```

### Step 2 — Record your gameplay

```bash
python record_gameplay.py
```

| Key | Action |
|-----|--------|
| `F8` | Start recording |
| `F9` | Stop and save |
| `ESC` | Quit |

Record at least 30–60 minutes of varied gameplay.

### Step 3 — Train

```bash
python train.py                  # full pipeline
python train.py --skip-pretrain  # skip Phase 1
python train.py --resume         # resume from checkpoint
```

Models saved to `models/best_model.pt` and `models/final_model.pt`.

| Hardware | ~10k frames | ETA |
|----------|-------------|-----|
| RTX 3060 | ~10k frames | ~20 min |
| CPU only | ~10k frames | ~3 hrs |

### Step 4 — Run the AI

```bash
python inference.py            # live inputs
python inference.py --dry-run  # predict only, no inputs
python inference.py --hud      # show live HUD window
```

| Key | Action |
|-----|--------|
| `F10` | Pause / Resume |
| `F11` | Toggle dry-run |
| `ESC` | Stop |

---

## Config Reference (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RECORD_FPS` | 10 | FPS during recording |
| `FRAME_W/H` | 320×180 | Frame resolution |
| `KEY_ACTIONS` | WASD+more | Keys tracked |
| `MOUSE_BINS` | 21 | Mouse discretisation bins |
| `NUM_STACK` | 2 | Frames stacked per input |
| `HIDDEN_DIM` | 512 | Model feature size |
| `BATCH_SIZE` | 32 | Training batch size |
| `NUM_EPOCHS` | 60 | Fine-tuning epochs |
| `INFERENCE_FPS` | 10 | AI decision rate |
| `KEY_CONFIDENCE_THRESH` | 0.45 | Key press threshold |
| `MOUSE_SCALE` | 1.0 | Mouse movement multiplier |

---

## How It Works

**Behavioural cloning** — imitation learning from your own play:

1. **Phase 1 (Pretraining)** — Inverse Dynamics Model watches pairs of video frames and learns to predict scene changes. Trains the visual encoder without needing labelled inputs.

2. **Phase 2 (Fine-tuning)** — The pretrained encoder transfers into `GameAI`. Trained on your recordings with exact labels (keys held, mouse moved). Four output heads: key presses, mouse X, mouse Y, clicks.

3. **Inference** — Model runs at ~10 FPS, stacks the last 2 frames for motion context, predicts an action, executes it via `pynput`.

Same fundamental approach as OpenAI's VPT for Minecraft, scaled for personal use.

---

## Project Structure

```
roblox_ai/
├── config.py              # All configuration
├── extract_frames.py      # Step 1: Extract video frames
├── record_gameplay.py     # Step 2: Record gameplay
├── build_dataset.py       # Dataset builders + action encoding
├── model.py               # Neural network architectures
├── train.py               # Step 3: Train
├── inference.py           # Step 4: AI plays live
├── gui.py                 # Web dashboard (localhost:5000)
├── start.py               # One-command launcher
├── requirements.txt
data/
├── videos/                # Video files
├── frames/                # Extracted frames (unlabelled)
├── recordings/            # Gameplay recordings (labelled)
models/                    # Saved model weights
```
