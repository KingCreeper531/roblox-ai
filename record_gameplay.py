# record_gameplay.py
# Records gameplay as synced (frame, action) pairs.
# Press F8 to start, F9 to stop, ESC to quit.

import os, sys, cv2, json, time, threading, argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import mss
from pynput import keyboard, mouse
from pynput.keyboard import Key, KeyCode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


class InputState:
    KEY_MAP = {
        'w': {'w'}, 'a': {'a'}, 's': {'s'}, 'd': {'d'},
        'space': {Key.space}, 'shift': {Key.shift, Key.shift_l, Key.shift_r},
        'e': {'e'}, 'f': {'f'}, 'q': {'q'}, 'r': {'r'},
    }

    def __init__(self):
        self._lock = threading.Lock()
        self._held_keys = set()
        self._mouse_pos = (0, 0)
        self._mouse_buttons = {'left': False, 'right': False}
        self._prev_pos = None

    def on_press(self, key):
        raw = key.char.lower() if isinstance(key, KeyCode) and key.char else key
        with self._lock: self._held_keys.add(raw)

    def on_release(self, key):
        raw = key.char.lower() if isinstance(key, KeyCode) and key.char else key
        with self._lock: self._held_keys.discard(raw)

    def on_move(self, x, y):
        with self._lock: self._mouse_pos = (x, y)

    def on_click(self, x, y, button, pressed):
        btn = 'left' if button == mouse.Button.left else 'right' if button == mouse.Button.right else None
        if btn:
            with self._lock: self._mouse_buttons[btn] = pressed

    def snapshot(self):
        with self._lock:
            held = set(self._held_keys)
            pos  = self._mouse_pos
            btns = dict(self._mouse_buttons)
        keys = {a: int(bool(held & t)) for a, t in self.KEY_MAP.items()}
        if self._prev_pos is None:
            dx, dy = 0, 0
        else:
            dx = int(np.clip(pos[0]-self._prev_pos[0], -config.MOUSE_MAX_DELTA, config.MOUSE_MAX_DELTA))
            dy = int(np.clip(pos[1]-self._prev_pos[1], -config.MOUSE_MAX_DELTA, config.MOUSE_MAX_DELTA))
        self._prev_pos = pos
        return {"keys": keys, "mouse": {"dx": dx, "dy": dy}, "clicks": btns}


class ScreenCapture:
    def __init__(self):
        self._sct = mss.mss()
        self._monitor = config.SCREEN_REGION or self._sct.monitors[1]

    def grab(self):
        img = self._sct.grab(self._monitor)
        frame = np.array(img)[:, :, :3]
        return cv2.resize(frame, (config.FRAME_W, config.FRAME_H), interpolation=cv2.INTER_AREA)


class SessionRecorder:
    def __init__(self, session_id, preview=False):
        self.session_id = session_id
        self.preview    = preview
        self.out_dir    = os.path.join(config.RECORDINGS_DIR, session_id)
        self.frames_dir = os.path.join(self.out_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self._state     = InputState()
        self._capture   = ScreenCapture()
        self._recording = False
        self._log       = []
        self._frame_idx = 0
        self._interval  = 1.0 / config.RECORD_FPS
        self._start_key = self._stop_key = self._quit_key = False

    def _on_kb_press(self, key):
        self._state.on_press(key)
        if key == Key.f8:  self._start_key = True
        elif key == Key.f9: self._stop_key  = True
        elif key == Key.esc: self._quit_key  = True

    def run(self):
        kb = keyboard.Listener(on_press=self._on_kb_press, on_release=self._state.on_release, suppress=False)
        ms = mouse.Listener(on_move=self._state.on_move, on_click=self._state.on_click)
        kb.start(); ms.start()
        print(f"\nGAMEPLAY RECORDER | Session: {self.session_id}")
        print("F8=Start  F9=Stop  ESC=Quit\n")
        try:
            while not self._quit_key:
                t0 = time.perf_counter()
                if self._start_key and not self._recording:
                    self._recording = True; self._start_key = False
                    print("Recording started...")
                if self._stop_key and self._recording:
                    self._recording = False; self._stop_key = False
                    print(f"Stopped. {self._frame_idx} frames.")
                    self._save(); break
                if self._recording:
                    frame  = self._capture.grab()
                    action = self._state.snapshot()
                    fname  = f"{self._frame_idx:07d}.jpg"
                    cv2.imwrite(os.path.join(self.frames_dir, fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    self._log.append({"frame": fname, "timestamp": time.time(), "action": action})
                    self._frame_idx += 1
                time.sleep(max(0.0, self._interval - (time.perf_counter() - t0)))
        except KeyboardInterrupt:
            pass
        finally:
            kb.stop(); ms.stop()

    def _save(self):
        meta = {"session_id": self.session_id, "fps": config.RECORD_FPS,
                "frame_w": config.FRAME_W, "frame_h": config.FRAME_H,
                "total_frames": len(self._log), "recorded_at": datetime.now().isoformat()}
        with open(os.path.join(self.out_dir, "meta.json"), "w") as f: json.dump(meta, f, indent=2)
        with open(os.path.join(self.out_dir, "actions.json"), "w") as f: json.dump(self._log, f, indent=2)
        print(f"Saved to {self.out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    session_id = args.session or datetime.now().strftime("session_%Y%m%d_%H%M%S")
    SessionRecorder(session_id, args.preview).run()

if __name__ == "__main__":
    main()
