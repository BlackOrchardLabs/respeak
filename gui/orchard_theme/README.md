
# Black Orchard Tkinter Theme Pack

This bundle provides a ready-to-drop-in visual theme for the Hermes TTS GUI.

## Structure

- `colors.py` — shared color constants.
- `tkinter_theme.py` — OrchardButton, waveform canvas, progress bar styling, window frame, and animator.
- `assets/buttons/` — PNG button backgrounds (idle/hover/active) for all roles.
- `assets/backgrounds/` — waveform HUD plate and concentric-ring backgrounds.
- `assets/icons/` — record/stop/play icons as simple flat PNGs.

## Basic Integration

1. Copy this folder into your project, e.g.

   ```
   your_project/
       tts_gui.py
       orchard_theme/
           __init__.py
           colors.py
           tkinter_theme.py
           assets/
   ```

2. In `orchard_theme/__init__.py`, you can export helpers (already safe to leave empty or add):

   ```python
   from .colors import *
   from .tkinter_theme import (
       OrchardButton,
       apply_orchard_window_frame,
       style_progressbar,
       create_progressbar,
       create_waveform_canvas,
       WaveformAnimator,
   )
   ```

3. In `tts_gui.py`:

   ```python
   from orchard_theme.tkinter_theme import (
       OrchardButton,
       apply_orchard_window_frame,
       style_progressbar,
       create_progressbar,
       create_waveform_canvas,
       WaveformAnimator,
   )
   ```

4. After creating the root window:

   ```python
   root = tk.Tk()
   root.geometry("600x550")
   apply_orchard_window_frame(root)
   style_progressbar(root)
   ```

5. When building your UI:

   - Use `OrchardButton` for buttons:

     ```python
     self.btn_record = OrchardButton(frame, text="● Record", role="record", command=self.start_recording)
     self.btn_stop = OrchardButton(frame, text="■ Stop", role="stop", command=self.stop_recording)
     self.btn_play_sample = OrchardButton(frame, text="Play Sample", role="play", command=self.play_sample)
     self.btn_rerecord = OrchardButton(frame, text="↺ Re-record", role="secondary", command=self.rerecord)
     self.btn_save = OrchardButton(frame, text="✓ Save sample", role="secondary", command=self.save_sample)
     self.btn_speak = OrchardButton(frame, text="▶ Speak", role="speak", command=self.speak_text)
     self.btn_clear = OrchardButton(frame, text="✖ Clear", role="secondary", command=self.clear_text)
     ```

   - Create the waveform canvas:

     ```python
     self.wave_canvas = create_waveform_canvas(record_frame, width=540, height=180)
     self.wave_canvas.grid(row=3, column=0, columnspan=3, pady=(8, 8))

     self.wave_animator = WaveformAnimator(self.wave_canvas, 540, 180)
     self.wave_animator.start()
     ```

   - Wire the animator to state changes:

     ```python
     def start_recording(self):
         # your record logic...
         self.wave_animator.set_state("recording")

     def stop_recording(self):
         # your stop logic...
         self.wave_animator.set_state("idle")

     def play_sample(self):
         # your playback logic...
         self.wave_animator.set_state("playback")
     ```

6. Progress bar:

   ```python
   self.progress = create_progressbar(record_frame)
   self.progress.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(4, 4))
   ```

All PNG assets are simple flat placeholders styled with the Orchard palette, so you can freely replace them later with higher-fidelity art without changing any code.
