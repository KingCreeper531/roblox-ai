# inference.py - Run the trained AI player live
# Usage: python inference.py | python inference.py --dry-run | python inference.py --hud
# Controls: F10=pause/resume  F11=toggle dry-run  ESC=stop

import os, sys, time, argparse, collections
import cv2, numpy as np, torch
import mss
from pynput import keyboard as kb_module
from pynput.keyboard import Key, Controller as KeyController
from pynput.mouse    import Controller as MouseController, Button

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from model         import GameAI
from build_dataset import frame_to_tensor, decode_action

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}\nRun train.py first.")
    model = GameAI().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


class InputExecutor:
    KEY_MAP = {'w':'w','a':'a','s':'s','d':'d','space':Key.space,'shift':Key.shift,'e':'e','f':'f','q':'q','r':'r'}

    def __init__(self):
        self._kb = KeyController(); self._ms = MouseController(); self._held = set()

    def execute(self, action, dry_run=False):
        if dry_run: return
        desired = {k for k, v in action["keys"].items() if v}
        for k in desired - self._held:
            try: self._kb.press(self.KEY_MAP.get(k, k))
            except: pass
        for k in self._held - desired:
            try: self._kb.release(self.KEY_MAP.get(k, k))
            except: pass
        self._held = desired
        dx = int(action["mouse"]["dx"] * config.MOUSE_SCALE)
        dy = int(action["mouse"]["dy"] * config.MOUSE_SCALE)
        if dx or dy:
            try: self._ms.move(dx, dy)
            except: pass
        try:
            if action["clicks"].get("left"):  self._ms.click(Button.left, 1)
            if action["clicks"].get("right"): self._ms.click(Button.right, 1)
        except: pass

    def release_all(self):
        for k in list(self._held):
            try: self._kb.release(self.KEY_MAP.get(k, k))
            except: pass
        self._held.clear()


def draw_hud(frame, action, fps, paused):
    img = cv2.resize(frame, (480, 270))
    overlay = img.copy()
    cv2.rectangle(overlay, (0,0), (img.shape[1], 60), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    status = "PAUSED" if paused else f"{fps:.1f} FPS"
    color  = (0,80,240) if paused else (0,220,80)
    cv2.putText(img, status, (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    active = " ".join(k.upper() for k,v in action["keys"].items() if v)
    cv2.putText(img, f"Keys: {active or '-'}", (8,45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1)
    dx, dy = action["mouse"]["dx"], action["mouse"]["dy"]
    cv2.putText(img, f"Mouse: ({dx:+d},{dy:+d})", (200,45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,220,255), 1)
    return img


class AIPlayer:
    def __init__(self, model_path, dry_run=False, show_hud=False):
        print(f"Loading model: {model_path}")
        self.model    = load_model(model_path)
        self.executor = InputExecutor()
        self.dry_run  = dry_run
        self.show_hud = show_hud
        self._paused   = False
        self._stop     = False
        self._frame_buf = collections.deque(maxlen=config.NUM_STACK)
        self._fps_win   = collections.deque(maxlen=20)
        print(f"Device: {DEVICE}  Dry-run: {dry_run}  HUD: {show_hud}")

    def _on_key(self, key):
        if key == Key.f10:
            self._paused = not self._paused
            if self._paused: self.executor.release_all()
            print("PAUSED" if self._paused else "RESUMED")
        elif key == Key.f11:
            self.dry_run = not self.dry_run
            print(f"Dry-run: {self.dry_run}")
        elif key == Key.esc:
            self._stop = True

    @torch.no_grad()
    def _predict(self, frame):
        t = frame_to_tensor(frame).unsqueeze(0)
        self._frame_buf.append(t)
        while len(self._frame_buf) < config.NUM_STACK:
            self._frame_buf.appendleft(self._frame_buf[0])
        stacked = torch.cat(list(self._frame_buf), dim=1).to(DEVICE)
        out = self.model(stacked)
        return decode_action(out["key_logits"][0], out["mouse_x_logits"][0], out["mouse_y_logits"][0], out["click_logits"][0])

    def run(self):
        listener = kb_module.Listener(on_press=self._on_key, suppress=False)
        listener.start()
        sct = mss.mss()
        monitor  = config.SCREEN_REGION or sct.monitors[1]
        interval = 1.0 / config.INFERENCE_FPS
        print("\nAI PLAYER ACTIVE | F10=Pause  F11=DryRun  ESC=Stop\n")
        action = {"keys": {k: False for k in config.KEY_ACTIONS}, "mouse": {"dx":0,"dy":0}, "clicks": {"left":False,"right":False}}
        try:
            while not self._stop:
                t0 = time.perf_counter()
                raw = sct.grab(monitor)
                frame = cv2.resize(np.array(raw)[:,:,:3], (config.FRAME_W, config.FRAME_H), interpolation=cv2.INTER_AREA)
                if not self._paused:
                    action = self._predict(frame)
                    self.executor.execute(action, dry_run=self.dry_run)
                elapsed = time.perf_counter() - t0
                self._fps_win.append(1.0/max(elapsed,1e-6))
                if self.show_hud:
                    fps = sum(self._fps_win)/len(self._fps_win)
                    cv2.imshow("AI HUD", draw_hud(frame, action, fps, self._paused))
                    if cv2.waitKey(1) & 0xFF == 27: self._stop = True
                time.sleep(max(0.0, interval - elapsed))
        except KeyboardInterrupt:
            pass
        finally:
            self.executor.release_all(); listener.stop()
            if self.show_hud: cv2.destroyAllWindows()
            print("AI stopped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default=os.path.join(config.MODELS_DIR, config.BEST_MODEL_NAME))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hud",     action="store_true")
    args = parser.parse_args()
    AIPlayer(args.model, args.dry_run, args.hud).run()

if __name__ == "__main__":
    main()
