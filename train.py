# train.py - Two-phase training: inverse dynamics pretraining + behavioural cloning
# Usage: python train.py | python train.py --skip-pretrain | python train.py --resume

import os, sys, argparse, time
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from model        import GameAI, InverseDynamicsNet
from build_dataset import build_labelled_dataset, build_pretrain_dataset

torch.manual_seed(config.SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"[Device] {DEVICE}")


def compute_bc_loss(out, batch):
    _, action = batch
    loss_keys   = F.binary_cross_entropy_with_logits(out["key_logits"], action["keys"].to(DEVICE),
                    pos_weight=torch.tensor([3.0]*len(config.KEY_ACTIONS)).to(DEVICE))
    loss_mouse  = 0.5*(F.cross_entropy(out["mouse_x_logits"], action["mouse_x_bin"].to(DEVICE)) +
                       F.cross_entropy(out["mouse_y_logits"], action["mouse_y_bin"].to(DEVICE)))
    loss_clicks = F.binary_cross_entropy_with_logits(out["click_logits"], action["clicks"].to(DEVICE),
                    pos_weight=torch.tensor([5.0]*len(config.MOUSE_BUTTONS)).to(DEVICE))
    total = loss_keys + loss_mouse + 0.5*loss_clicks
    return total, {"keys": loss_keys.item(), "mouse": loss_mouse.item(), "clicks": loss_clicks.item(), "total": total.item()}


@torch.no_grad()
def batch_accuracy(out, action):
    key_acc   = ((torch.sigmoid(out["key_logits"]) > config.KEY_CONFIDENCE_THRESH) == action["keys"].to(DEVICE).bool()).float().mean().item()
    mouse_acc = (out["mouse_x_logits"].argmax(-1) == action["mouse_x_bin"].to(DEVICE)).float().mean().item()
    return {"key_acc": key_acc, "mouse_acc": mouse_acc}


def pretrain(num_epochs=config.PRETRAIN_EPOCHS):
    print("\nPHASE 1 - Inverse Dynamics Pretraining")
    ds = build_pretrain_dataset(augment=True)
    if ds is None or len(ds) == 0:
        print("  No frames found. Skipping.\n"); return None
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    idm    = InverseDynamicsNet().to(DEVICE)
    opt    = torch.optim.AdamW(idm.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    sched  = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config.LEARNING_RATE*5, steps_per_epoch=len(loader), epochs=num_epochs)
    best   = float("inf")
    for epoch in range(1, num_epochs+1):
        idm.train(); running = 0.0
        for frames, diff in tqdm(loader, desc=f"  Pretrain {epoch}/{num_epochs}", leave=False):
            opt.zero_grad()
            loss = F.mse_loss(idm(frames.to(DEVICE)), diff.to(DEVICE))
            loss.backward(); torch.nn.utils.clip_grad_norm_(idm.parameters(), config.GRAD_CLIP); opt.step(); sched.step()
            running += loss.item()
        avg = running/len(loader)
        print(f"  Epoch {epoch:3d}  loss={avg:.5f}")
        if avg < best:
            best = avg; torch.save(idm.state_dict(), os.path.join(config.MODELS_DIR, "pretrain_best.pt"))
    print(f"  Pretraining done. Best: {best:.5f}")
    return idm


def finetune(pretrained_idm=None, num_epochs=config.NUM_EPOCHS, resume=False):
    print("\nPHASE 2 - Behavioural Cloning")
    train_ds, val_ds = build_labelled_dataset(augment_train=True)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    model = GameAI().to(DEVICE)
    if pretrained_idm:
        model.encoder.load_state_dict(pretrained_idm.encoder.state_dict()); print("  Encoder transferred.")
    start_epoch = 1; best_val = float("inf")
    ckpt_path = os.path.join(config.MODELS_DIR, "checkpoint.pt")
    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"]); start_epoch = ckpt["epoch"]+1; best_val = ckpt.get("best_val", float("inf"))
    opt   = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs, eta_min=1e-6)
    print(f"  Params: {model.n_params():,}  Train: {len(train_ds)}  Val: {len(val_ds)}  Epochs: {num_epochs}\n")
    for epoch in range(start_epoch, num_epochs+1):
        model.train(); tl = {"total":0,"keys":0,"mouse":0,"clicks":0}; ta = {"key_acc":0,"mouse_acc":0}; nb = 0
        for batch in tqdm(train_loader, desc=f"  Train {epoch}/{num_epochs}", leave=False):
            frames, action = batch; frames = frames.to(DEVICE)
            opt.zero_grad(); out = model(frames); loss, comps = compute_bc_loss(out, batch)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP); opt.step()
            for k in tl: tl[k] += comps[k]
            for k in ta: ta[k] += batch_accuracy(out, action)[k]
            nb += 1
        model.eval(); vl = {"total":0,"keys":0,"mouse":0,"clicks":0}; va = {"key_acc":0,"mouse_acc":0}; nv = 0
        with torch.no_grad():
            for batch in val_loader:
                frames, action = batch; out = model(frames.to(DEVICE)); _, comps = compute_bc_loss(out, batch); acc = batch_accuracy(out, action)
                for k in vl: vl[k] += comps[k]
                for k in va: va[k] += acc[k]
                nv += 1
        sched.step()
        print(f"  Ep {epoch:3d}  train={tl['total']/nb:.4f}  val={vl['total']/nv:.4f}  key_acc={va['key_acc']/nv:.2%}  lr={sched.get_last_lr()[0]:.2e}")
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "best_val": best_val}, ckpt_path)
        if vl["total"]/nv < best_val:
            best_val = vl["total"]/nv; torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, config.BEST_MODEL_NAME))
            print(f"  New best: {best_val:.5f} (saved)")
        if epoch % config.CHECKPOINT_EVERY == 0:
            torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, f"epoch_{epoch:04d}.pt"))
    torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, config.FINAL_MODEL_NAME))
    print(f"\nTraining complete. Best: {best_val:.5f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pretrain",   action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=config.PRETRAIN_EPOCHS)
    parser.add_argument("--epochs",          type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--resume",          action="store_true")
    args = parser.parse_args()
    t0  = time.time()
    idm = None if args.skip_pretrain else pretrain(args.pretrain_epochs)
    finetune(idm, args.epochs, args.resume)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
