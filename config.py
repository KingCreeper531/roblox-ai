import os

ROOT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(ROOT_DIR, "data")
VIDEOS_DIR     = os.path.join(DATA_DIR, "videos")
FRAMES_DIR     = os.path.join(DATA_DIR, "frames")
RECORDINGS_DIR = os.path.join(DATA_DIR, "recordings")
DATASET_DIR    = os.path.join(DATA_DIR, "dataset")
MODELS_DIR     = os.path.join(ROOT_DIR, "models")

for d in [VIDEOS_DIR, FRAMES_DIR, RECORDINGS_DIR, DATASET_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

EXTRACT_FPS   = 10
RECORD_FPS    = 10
FRAME_W       = 320
FRAME_H       = 180
SCREEN_REGION = None

KEY_ACTIONS     = ['w', 'a', 's', 'd', 'space', 'shift', 'e', 'f', 'q', 'r']
MOUSE_MAX_DELTA = 60
MOUSE_BINS      = 21
MOUSE_BUTTONS   = ['left', 'right']

NUM_STACK   = 2
IN_CHANNELS = NUM_STACK * 3
HIDDEN_DIM  = 512
DROPOUT     = 0.3

BATCH_SIZE       = 32
LEARNING_RATE    = 3e-4
WEIGHT_DECAY     = 1e-4
NUM_EPOCHS       = 60
PRETRAIN_EPOCHS  = 20
TRAIN_SPLIT      = 0.85
GRAD_CLIP        = 1.0
CHECKPOINT_EVERY = 5
BEST_MODEL_NAME  = "best_model.pt"
FINAL_MODEL_NAME = "final_model.pt"

INFERENCE_FPS           = 10
KEY_CONFIDENCE_THRESH   = 0.45
CLICK_CONFIDENCE_THRESH = 0.50
MOUSE_SCALE             = 1.0

SEED = 42
