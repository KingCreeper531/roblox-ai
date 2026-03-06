# build_dataset.py - PyTorch Dataset builders and action encoding helpers

import os, sys, json, random
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import Dataset, ConcatDataset, random_split
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)


def encode_action(action):
    keys = [float(action["keys"].get(k, 0)) for k in config.KEY_ACTIONS]
    def delta_to_bin(d):
        half = config.MOUSE_MAX_DELTA; bins = config.MOUSE_BINS
        return max(0, min(bins-1, int(((max(-half,min(half,d))+half)/(2*half))*(bins-1)+0.5)))
    mx = delta_to_bin(action["mouse"]["dx"])
    my = delta_to_bin(action["mouse"]["dy"])
    clicks = [float(action["clicks"].get(b, False)) for b in config.MOUSE_BUTTONS]
    return {"keys": torch.tensor(keys, dtype=torch.float32),
            "mouse_x_bin": torch.tensor(mx, dtype=torch.long),
            "mouse_y_bin": torch.tensor(my, dtype=torch.long),
            "clicks": torch.tensor(clicks, dtype=torch.float32)}


def decode_action(keys_tensor, mouse_x_bin, mouse_y_bin, clicks_tensor,
                  key_thresh=config.KEY_CONFIDENCE_THRESH,
                  click_thresh=config.CLICK_CONFIDENCE_THRESH):
    def bin_to_delta(b):
        half = config.MOUSE_MAX_DELTA; bins = config.MOUSE_BINS
        return int((b/(bins-1))*2*half - half)
    keys_probs  = torch.sigmoid(keys_tensor).cpu().numpy()
    click_probs = torch.sigmoid(clicks_tensor).cpu().numpy()
    mx_bin = int(mouse_x_bin.argmax().item()) if mouse_x_bin.dim() > 0 else int(mouse_x_bin.item())
    my_bin = int(mouse_y_bin.argmax().item()) if mouse_y_bin.dim() > 0 else int(mouse_y_bin.item())
    return {"keys": {k: bool(keys_probs[i] > key_thresh) for i, k in enumerate(config.KEY_ACTIONS)},
            "mouse": {"dx": bin_to_delta(mx_bin), "dy": bin_to_delta(my_bin)},
            "clicks": {b: bool(click_probs[i] > click_thresh) for i, b in enumerate(config.MOUSE_BUTTONS)}}


_norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

def frame_to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _norm(torch.from_numpy(rgb).permute(2,0,1).float()/255.0)

def augment_frame(t):
    if random.random() < 0.4:
        t = torch.clamp(t * random.uniform(0.8, 1.2), 0, 1)
    return t


class LabelledDataset(Dataset):
    def __init__(self, session_dir, augment=False):
        self.frames_dir = os.path.join(session_dir, "frames")
        self.augment    = augment
        with open(os.path.join(session_dir, "actions.json")) as f:
            self.log = json.load(f)
        self.frame_files = sorted([e["frame"] for e in self.log], key=lambda x: int(Path(x).stem))
        self.entries     = self.log

    def __len__(self): return max(0, len(self.frame_files) - (config.NUM_STACK - 1))

    def __getitem__(self, idx):
        real = idx + config.NUM_STACK - 1
        frames = []
        for i in range(config.NUM_STACK):
            fi  = real - (config.NUM_STACK - 1 - i)
            img = cv2.imread(os.path.join(self.frames_dir, self.frame_files[fi]))
            if img is None: img = np.zeros((config.FRAME_H, config.FRAME_W, 3), dtype=np.uint8)
            t = frame_to_tensor(img)
            if self.augment: t = augment_frame(t)
            frames.append(t)
        return torch.cat(frames, dim=0), encode_action(self.entries[real]["action"])


class InverseDynamicsDataset(Dataset):
    def __init__(self, frames_root, augment=False):
        self.augment = augment; self.pairs = []
        root    = Path(frames_root)
        subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
        for src in (subdirs or [root]):
            flist = sorted(src.glob("*.jpg"), key=lambda p: int(p.stem))
            for i in range(len(flist)-1):
                self.pairs.append((str(flist[i]), str(flist[i+1])))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        def load(p):
            img = cv2.imread(p)
            if img is None: img = np.zeros((config.FRAME_H, config.FRAME_W, 3), dtype=np.uint8)
            t = frame_to_tensor(img)
            return augment_frame(t) if self.augment else t
        t0, t1 = load(self.pairs[idx][0]), load(self.pairs[idx][1])
        return torch.cat([t0, t1], dim=0), torch.tensor(torch.mean(torch.abs(t1-t0)).item(), dtype=torch.float32)


class AugmentedSubset(Dataset):
    def __init__(self, subset, augment=False): self.subset = subset; self.augment = augment
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        frames, action = self.subset[idx]
        if self.augment:
            frames = torch.cat([augment_frame(c) for c in torch.split(frames, 3, dim=0)], dim=0)
        return frames, action


def build_labelled_dataset(augment_train=True):
    sessions = [d for d in Path(config.RECORDINGS_DIR).iterdir()
                if d.is_dir() and (d/"actions.json").exists() and (d/"frames").exists()]
    if not sessions: raise RuntimeError(f"No sessions in {config.RECORDINGS_DIR}. Run record_gameplay.py first.")
    all_ds = []
    for sess in sessions:
        ds = LabelledDataset(str(sess))
        if len(ds) > 0: all_ds.append(ds); print(f"  {sess.name}: {len(ds)} samples")
    if not all_ds: raise RuntimeError("All sessions produced 0 samples.")
    full = ConcatDataset(all_ds)
    n_train = int(len(full)*config.TRAIN_SPLIT); n_val = len(full)-n_train
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(config.SEED))
    print(f"  Train: {n_train}  Val: {n_val}")
    return AugmentedSubset(train_ds, augment=augment_train), val_ds


def build_pretrain_dataset(augment=True):
    root = Path(config.FRAMES_DIR)
    if not root.exists() or not any(root.iterdir()):
        print("  No video frames found. Skipping pretraining."); return None
    ds = InverseDynamicsDataset(str(root), augment=augment)
    print(f"  Video frame pairs: {len(ds)}"); return ds


if __name__ == "__main__":
    print("Dataset Inspector")
    try:
        tr, va = build_labelled_dataset(False)
        print(f"Train: {len(tr)}  Val: {len(va)}")
    except RuntimeError as e: print(f"  {e}")
    pt = build_pretrain_dataset(False)
    if pt: print(f"Pretrain pairs: {len(pt)}")
    print("Done.")
