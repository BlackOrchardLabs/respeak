#!/usr/bin/env python3
# TTS GUI - Hermes Voice Kernel Studio (Orchard UI)
# Merged version: preserves original TTS logic (Coqui XTTS + pyttsx3 fallback)
# but wraps it in the holographic Orchard cobalt UI.
#
# Modified for voice-kernel-tts repo with kernel selector dropdown.

import os
import sys
import json
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import threading
import tempfile
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile

# === Orchard theme imports ===
from orchard_theme.tkinter_theme import (
    OrchardButton,
    apply_orchard_window_frame,
    style_progressbar,
    create_progressbar,
    create_waveform_canvas,
    WaveformAnimator,
)

from orchard_theme.colors import (
    BG_MAIN, BG_PANEL, BG_PANEL_ALT, BG_INPUT,
    TEXT_PRIMARY, TEXT_SUBTLE,
)

# === Voice kernel path setup ===
# We're in gui/ subfolder, so parent is the repo root
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Folders for kernels and samples
KERNELS_DIR = REPO_ROOT / "kernels"
SAMPLES_DIR = REPO_ROOT / "samples"


def scan_kernels():
    """Scan kernels/ folder and return list of (name, path) tuples"""
    kernels = []
    if KERNELS_DIR.exists():
        for kernel_file in KERNELS_DIR.glob("*.json"):
            try:
                with open(kernel_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    name = data.get("kernel_name", kernel_file.stem)
                    kernels.append((name, kernel_file))
            except Exception as e:
                print(f"Warning: Could not parse {kernel_file}: {e}")
    return kernels


def get_speaker_wav_from_kernel(kernel_path):
    """Read speaker_wav field from kernel JSON and resolve to absolute path"""
    try:
        with open(kernel_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            speaker_wav = data.get("speaker_wav", "")
            if speaker_wav:
                # Resolve relative to repo root
                wav_path = REPO_ROOT / speaker_wav
                if wav_path.exists():
                    return wav_path
                # Try samples/ folder directly
                wav_path = SAMPLES_DIR / Path(speaker_wav).name
                if wav_path.exists():
                    return wav_path
    except Exception as e:
        print(f"Warning: Could not read speaker_wav from {kernel_path}: {e}")
    return None


class TTSApp(tk.Tk):
    """
    Hermes TTS UI with Orchard holographic theme.
    Features:
      - Kernel selector dropdown (scans kernels/ folder)
      - Coqui XTTS voice kernel synthesis
      - pyttsx3 system TTS fallback
      - Recording with waveform visualization
    """

    def __init__(self):
        super().__init__()

        # ===== WINDOW SETUP =====
        self.title("Re:speak - Voice Kernel Studio")
        self.geometry("620x580")
        self.minsize(620, 580)

        # Detect which TTS mode to use
        self.tts_mode = "system"  # default
        self.tts_engine = None
        self.voice_kernel = None
        self.counter = 1
        self.is_speaking = False

        # Kernel selection
        self.available_kernels = scan_kernels()
        self.selected_kernel_path = None
        self.selected_sample_path = None

        # State variables for UI
        self.word_count_var = tk.StringVar(value="Words: 0 / 1000")
        self.status_var = tk.StringVar(value="Initializing...")
        self.current_kernel_name = None

        # Recording state variables
        self.is_recording = False
        self.recording_data = []
        self.recording_stream = None
        self.recording_thread = None
        self.sample_rate = 22050  # Sample rate for voice kernel compatibility
        self.recorded_audio = None  # Stores the final recorded audio

        # Apply Orchard styling
        apply_orchard_window_frame(self)
        style_progressbar(self)

        # Main container
        self.main_frame = tk.Frame(self, bg=BG_MAIN, bd=0, highlightthickness=0)
        self.main_frame.pack(fill="both", expand=True, padx=12, pady=12)

        # Build UI layout
        self._build_layout()

        # Initialize TTS with first available kernel (or system TTS)
        self._initialize_tts()
        self.status_var.set(self._get_status_text())

    def _apply_dark_titlebar(self):
        """Enable dark title bar on Windows 10/11"""
        try:
            import ctypes

            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            if hwnd == 0:
                hwnd = self.winfo_id()

            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19

            value = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                ctypes.byref(value), ctypes.sizeof(value)
            )
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1,
                ctypes.byref(value), ctypes.sizeof(value)
            )
            self.update()
        except Exception:
            pass

    # =========================
    #   LAYOUT
    # =========================
    def _build_layout(self):
        self._build_kernel_selector()
        self._build_recording_section()
        self._build_tts_section()

    def _build_kernel_selector(self):
        """Build kernel selection dropdown"""
        selector_frame = tk.Frame(self.main_frame, bg=BG_PANEL, bd=0, highlightthickness=0)
        selector_frame.pack(fill="x", pady=(0, 8))

        tk.Label(
            selector_frame,
            text="Voice Kernel:",
            bg=BG_PANEL,
            fg=TEXT_PRIMARY,
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left", padx=(6, 8))

        # Build kernel names list
        if self.available_kernels:
            kernel_names = [name for name, path in self.available_kernels]
        else:
            kernel_names = ["(no kernels found)"]

        self.kernel_var = tk.StringVar(value=kernel_names[0] if kernel_names else "")
        self.kernel_dropdown = ttk.Combobox(
            selector_frame,
            textvariable=self.kernel_var,
            values=kernel_names,
            width=35,
            state="readonly",
        )
        self.kernel_dropdown.pack(side="left", padx=(0, 8))
        self.kernel_dropdown.bind("<<ComboboxSelected>>", self._on_kernel_selected)

        # Refresh button
        self.btn_refresh = OrchardButton(
            selector_frame, text="Refresh", role="secondary",
            command=self._refresh_kernels
        )
        self.btn_refresh.pack(side="left", padx=(4, 4))

        # Status label (right side)
        self.kernel_status_label = tk.Label(
            selector_frame,
            text="",
            bg=BG_PANEL,
            fg=TEXT_SUBTLE,
            font=("Segoe UI", 9),
        )
        self.kernel_status_label.pack(side="right", padx=(4, 6))

    def _build_recording_section(self):
        rec_frame = tk.Frame(self.main_frame, bg=BG_PANEL, bd=0, highlightthickness=0)
        rec_frame.pack(fill="x", pady=(0, 10))

        # --- Row 0: Time selector + Status label ---
        top_row = tk.Frame(rec_frame, bg=BG_PANEL)
        top_row.pack(fill="x", pady=(4, 4))

        tk.Label(
            top_row, text="Time limit:", bg=BG_PANEL, fg=TEXT_SUBTLE,
            font=("Segoe UI", 9),
        ).pack(side="left", padx=(4, 2))

        self.time_var = tk.StringVar(value="30s")
        self.time_menu = ttk.Combobox(
            top_row, textvariable=self.time_var,
            values=["30s", "1min", "2min", "5min", "10min"],
            width=7, state="readonly",
        )
        self.time_menu.pack(side="left")

        tk.Label(top_row, bg=BG_PANEL, text="").pack(side="left", padx=12)

        tk.Label(
            top_row, textvariable=self.status_var, bg=BG_PANEL, fg=TEXT_PRIMARY,
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=(4, 4))

        # --- Row 1: Record / Stop / Play Sample buttons ---
        btn_row = tk.Frame(rec_frame, bg=BG_PANEL)
        btn_row.pack(fill="x", pady=(4, 4))

        self.btn_record = OrchardButton(
            btn_row, text="Record", role="record",
            command=self._on_record_click
        )
        self.btn_record.pack(side="left", padx=(4, 4))

        self.btn_stop_rec = OrchardButton(
            btn_row, text="Stop", role="stop",
            command=self._on_stop_record_click
        )
        self.btn_stop_rec.pack(side="left", padx=(4, 4))

        self.btn_play_sample = OrchardButton(
            btn_row, text="Play Sample", role="play",
            command=self._on_play_sample_click
        )
        self.btn_play_sample.pack(side="left", padx=(4, 4))

        # --- Row 2: Progress bar ---
        pb_row = tk.Frame(rec_frame, bg=BG_PANEL)
        pb_row.pack(fill="x", pady=(2, 4))

        self.progress = create_progressbar(pb_row)
        self.progress.pack(fill="x", padx=(4, 4))

        # --- Row 3: Waveform visualizer ---
        wave_row = tk.Frame(rec_frame, bg=BG_PANEL)
        wave_row.pack(fill="x", pady=(6, 4))

        self.wave_canvas = create_waveform_canvas(wave_row, width=560, height=140)
        self.wave_canvas.pack(padx=4, pady=(0, 4))

        self.wave_animator = WaveformAnimator(self.wave_canvas, 560, 140)
        self.wave_animator.set_state("idle")
        self.wave_animator.start()

        # --- Row 4: Re-record / Save sample ---
        bottom_row = tk.Frame(rec_frame, bg=BG_PANEL)
        bottom_row.pack(fill="x", pady=(4, 4))

        self.btn_rerecord = OrchardButton(
            bottom_row, text="Re-record", role="secondary",
            command=self._on_rerecord_click
        )
        self.btn_rerecord.pack(side="left", padx=(4, 4))

        self.btn_save = OrchardButton(
            bottom_row, text="Save sample", role="secondary",
            command=self._on_save_sample_click
        )
        self.btn_save.pack(side="left", padx=(4, 4))

    def _build_tts_section(self):
        tts_outer = tk.Frame(self.main_frame, bg=BG_PANEL_ALT, bd=0, highlightthickness=0)
        tts_outer.pack(fill="both", expand=True, pady=(0, 0))

        # Label row
        label_row = tk.Frame(tts_outer, bg=BG_PANEL_ALT)
        label_row.pack(fill="x", pady=(4, 2))

        tk.Label(
            label_row, text="TTS Input", bg=BG_PANEL_ALT, fg=TEXT_PRIMARY,
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left", padx=(6, 4))

        tk.Label(
            label_row, textvariable=self.word_count_var, bg=BG_PANEL_ALT,
            fg=TEXT_SUBTLE, font=("Segoe UI", 9),
        ).pack(side="right", padx=(4, 6))

        # Text box + scrollbar
        text_frame = tk.Frame(tts_outer, bg=BG_PANEL_ALT)
        text_frame.pack(fill="both", expand=True, padx=6, pady=(0, 4))

        self.tts_text = tk.Text(
            text_frame, bg="#0A1628", fg=TEXT_PRIMARY,
            insertbackground="#0099FF", relief=tk.FLAT, bd=0,
            height=6, wrap="word", font=("Segoe UI", 10),
        )
        self.tts_text.pack(fill="both", expand=True, side="left")
        self.tts_text.bind("<<Modified>>", self._on_text_modified)

        scrollbar = tk.Scrollbar(text_frame, command=self.tts_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.tts_text.config(yscrollcommand=scrollbar.set)

        # Button row
        btn_row = tk.Frame(tts_outer, bg=BG_PANEL_ALT)
        btn_row.pack(fill="x", pady=(4, 6))

        self.btn_speak = OrchardButton(
            btn_row, text="Speak", role="speak",
            command=self._on_speak_click
        )
        self.btn_speak.pack(side="left", padx=(4, 4))

        self.btn_stop_tts = OrchardButton(
            btn_row, text="Stop", role="stop",
            command=self._on_stop_tts_click
        )
        self.btn_stop_tts.pack(side="left", padx=(4, 4))
        self.btn_stop_tts.config(state=tk.DISABLED)

        self.btn_clear = OrchardButton(
            btn_row, text="Clear", role="secondary",
            command=self._on_clear_click
        )
        self.btn_clear.pack(side="left", padx=(4, 4))

    # =========================
    #   KERNEL SELECTION
    # =========================
    def _refresh_kernels(self):
        """Refresh the kernel list from disk"""
        self.available_kernels = scan_kernels()
        if self.available_kernels:
            kernel_names = [name for name, path in self.available_kernels]
            self.kernel_dropdown['values'] = kernel_names
            self.status_var.set(f"Found {len(self.available_kernels)} kernel(s)")
        else:
            self.kernel_dropdown['values'] = ["(no kernels found)"]
            self.status_var.set("No kernels found in kernels/")

    def _on_kernel_selected(self, event=None):
        """Handle kernel selection change"""
        selected_name = self.kernel_var.get()

        # Find the kernel path
        for name, path in self.available_kernels:
            if name == selected_name:
                self.selected_kernel_path = path
                self.selected_sample_path = get_speaker_wav_from_kernel(path)
                break

        # Reinitialize TTS with new kernel
        self.status_var.set(f"Loading {selected_name}...")
        self.update()
        self._initialize_tts()
        self.status_var.set(self._get_status_text())

    # =========================
    #   TTS INITIALIZATION
    # =========================
    def _initialize_tts(self):
        """Initialize TTS engine - try voice kernel first, fallback to system TTS"""

        # If no kernel selected, try to select first available
        if not self.selected_kernel_path and self.available_kernels:
            name, path = self.available_kernels[0]
            self.selected_kernel_path = path
            self.selected_sample_path = get_speaker_wav_from_kernel(path)
            self.kernel_var.set(name)

        # Check if we have valid kernel and sample
        if self.selected_kernel_path and self.selected_sample_path:
            if self.selected_kernel_path.exists() and self.selected_sample_path.exists():
                try:
                    os.environ["COQUI_TOS_AGREED"] = "1"

                    import torch
                    from torch.serialization import add_safe_globals
                    from TTS.tts.configs.xtts_config import XttsConfig
                    from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
                    from TTS.config.shared_configs import BaseDatasetConfig
                    from libs.kernel_loader import load_kernel
                    from TTS.api import TTS
                    import soundfile as sf

                    add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

                    # Patch TTS audio loading
                    def load_audio_patch(audiopath, load_sr=22050):
                        import torch
                        import soundfile as sf
                        try:
                            import torchaudio
                            use_torchaudio = True
                        except ImportError:
                            use_torchaudio = False

                        audio, sr = sf.read(audiopath)
                        audio_tensor = torch.FloatTensor(audio)

                        if audio_tensor.ndim > 1:
                            audio_tensor = audio_tensor.mean(dim=1)
                        if audio_tensor.ndim == 1:
                            audio_tensor = audio_tensor.unsqueeze(0)

                        if sr != load_sr:
                            if use_torchaudio:
                                audio_tensor = torchaudio.functional.resample(
                                    audio_tensor, orig_freq=sr, new_freq=load_sr
                                )
                            else:
                                import numpy as np
                                ratio = load_sr / sr
                                num_out = int(audio_tensor.shape[-1] * ratio)
                                audio_np = audio_tensor.squeeze(0).numpy()
                                x_old = np.linspace(0, 1, audio_np.shape[-1])
                                x_new = np.linspace(0, 1, num_out)
                                resampled = np.interp(x_new, x_old, audio_np).astype("float32")
                                audio_tensor = torch.from_numpy(resampled).unsqueeze(0)

                        return audio_tensor

                    import TTS.tts.models.xtts as xtts_module
                    xtts_module.load_audio = load_audio_patch

                    # Load voice kernel
                    print(f"Loading voice kernel from {self.selected_kernel_path}...")
                    self.voice_kernel = load_kernel(
                        str(self.selected_kernel_path),
                        audio_path=str(self.selected_sample_path)
                    )
                    print(f"Loaded kernel: {self.voice_kernel.name}")

                    # Initialize Coqui TTS (only if not already initialized)
                    if self.tts_engine is None or self.tts_mode != "voice_kernel":
                        print("Initializing Coqui XTTS v2...")
                        self.tts_engine = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
                        print("Voice Kernel TTS ready!")

                    self.tts_mode = "voice_kernel"
                    self.current_kernel_name = self.voice_kernel.name
                    self.kernel_status_label.config(text=f"Sample: {self.selected_sample_path.name}")
                    return

                except Exception as e:
                    print(f"Failed to initialize Voice Kernel: {e}")
                    import traceback
                    traceback.print_exc()

        # Fallback to system TTS
        print("Using system TTS (pyttsx3)...")
        self._init_system_tts()

    def _init_system_tts(self):
        """Initialize pyttsx3 system TTS"""
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_mode = "system"
            self.current_kernel_name = None
            self.kernel_status_label.config(text="System voice")
        except Exception as e:
            print(f"Failed to initialize system TTS: {e}")
            self.tts_mode = "none"
            self.kernel_status_label.config(text="No TTS available")

    def _get_status_text(self):
        """Get status text based on TTS mode"""
        if self.tts_mode == "voice_kernel":
            return f"Voice Kernel: {self.current_kernel_name}"
        elif self.tts_mode == "system":
            return "System TTS (no kernel loaded)"
        else:
            return "TTS unavailable"

    # =========================
    #   WORD COUNT
    # =========================
    def count_words(self, text: str) -> int:
        return len(text.split())

    def _on_text_modified(self, event=None):
        if self.tts_text.edit_modified():
            text = self.tts_text.get("1.0", "end").strip()
            word_count = self.count_words(text)
            self.word_count_var.set(f"Words: {word_count} / 1000")
            self.tts_text.edit_modified(False)

    # =========================
    #   SPEAK / STOP / CLEAR
    # =========================
    def _on_speak_click(self):
        text = self.tts_text.get("1.0", "end").strip()

        if not text:
            messagebox.showwarning("No Text", "Please enter some text to speak.")
            return

        word_count = self.count_words(text)
        if word_count > 1000:
            messagebox.showerror(
                "Text Too Long",
                f"Text contains {word_count} words. Maximum is 1000 words."
            )
            return

        if self.tts_mode == "voice_kernel":
            thread = threading.Thread(target=self._speak_voice_kernel, args=(text,), daemon=True)
        elif self.tts_mode == "system":
            thread = threading.Thread(target=self._speak_system, args=(text,), daemon=True)
        else:
            messagebox.showerror("TTS Error", "No TTS engine available")
            return

        thread.start()

    def _speak_voice_kernel(self, text):
        """Speak using Voice Kernel (Coqui XTTS)"""
        try:
            self.is_speaking = True
            self._set_tts_buttons_busy(True)
            self.status_var.set("Generating speech...")
            self.wave_animator.set_state("playback")

            output_file = Path(tempfile.gettempdir()) / f"respeak_tts_{self.counter:03d}.wav"
            self.counter += 1

            print(f"Generating speech with Voice Kernel...")
            self.tts_engine.tts_to_file(
                text=text,
                file_path=str(output_file),
                speaker_wav=self.voice_kernel.get_speaker_wav(),
                language="en"
            )

            print(f"Playing: {output_file}")
            os.system(f'start "" "{output_file}"')

        except Exception as e:
            messagebox.showerror("TTS Error", f"Error during speech: {str(e)}")
        finally:
            self.is_speaking = False
            self._set_tts_buttons_busy(False)
            self.wave_animator.set_state("idle")
            self.status_var.set(self._get_status_text())

    def _speak_system(self, text):
        """Speak using system TTS (pyttsx3)"""
        try:
            self.is_speaking = True
            self._set_tts_buttons_busy(True)
            self.status_var.set("Speaking (system TTS)...")
            self.wave_animator.set_state("playback")

            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

        except Exception as e:
            messagebox.showerror("TTS Error", f"Error during speech: {str(e)}")
        finally:
            self.is_speaking = False
            self._set_tts_buttons_busy(False)
            self.wave_animator.set_state("idle")
            self.status_var.set(self._get_status_text())

    def _set_tts_buttons_busy(self, busy: bool):
        if busy:
            self.btn_speak.config(state=tk.DISABLED)
            self.btn_stop_tts.config(state=tk.NORMAL)
        else:
            self.btn_speak.config(state=tk.NORMAL)
            self.btn_stop_tts.config(state=tk.DISABLED)

    def _on_stop_tts_click(self):
        self.stop_speaking()

    def stop_speaking(self):
        """Stop current speech"""
        if self.is_speaking:
            if self.tts_mode == "system":
                self.tts_engine.stop()
            self.is_speaking = False
            self._set_tts_buttons_busy(False)
            self.wave_animator.set_state("idle")
            self.status_var.set("Playback stopped")

    def _on_clear_click(self):
        self.tts_text.delete("1.0", "end")
        self.word_count_var.set("Words: 0 / 1000")

    # =========================
    #   RECORDING
    # =========================
    def _parse_time_limit(self):
        time_str = self.time_var.get()
        if time_str.endswith("s"):
            return int(time_str[:-1])
        elif time_str.endswith("min"):
            return int(time_str[:-3]) * 60
        return 30

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
        self.recording_data.append(indata.copy())
        rms = np.sqrt(np.mean(indata**2))
        amplitude = min(1.0, rms * 10)
        self.after(0, lambda: self.wave_animator.update_with_amplitude(amplitude))

    def _recording_worker(self, duration_seconds):
        try:
            self.recording_data = []
            with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self._audio_callback):
                steps = 100
                for i in range(steps + 1):
                    if not self.is_recording:
                        break
                    progress = (i / steps) * 100
                    self.after(0, lambda p=progress: self.progress.configure(value=p))
                    threading.Event().wait(duration_seconds / steps)

            if self.recording_data:
                self.recorded_audio = np.concatenate(self.recording_data, axis=0)
                self.after(0, self._recording_complete)
            else:
                self.after(0, lambda: self.status_var.set("Recording failed - no data"))
        except Exception as e:
            self.after(0, lambda: self.status_var.set(f"Recording error: {str(e)}"))
            self.after(0, lambda: messagebox.showerror("Recording Error", str(e)))
        finally:
            self.is_recording = False
            self.after(0, lambda: self.wave_animator.set_state("idle"))

    def _recording_complete(self):
        self.status_var.set("Recording complete - click Save to store")
        self.wave_animator.set_state("idle")
        self.btn_record.config(state=tk.NORMAL)
        self.btn_stop_rec.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.NORMAL)
        self.progress["value"] = 100

    def _on_record_click(self):
        if self.is_recording:
            return
        duration = self._parse_time_limit()
        self.is_recording = True
        self.recorded_audio = None
        self.status_var.set(f"Recording for {self.time_var.get()}...")
        self.wave_animator.set_state("recording")
        self.progress["value"] = 0
        self.btn_record.config(state=tk.DISABLED)
        self.btn_stop_rec.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.DISABLED)
        self.recording_thread = threading.Thread(
            target=self._recording_worker, args=(duration,), daemon=True
        )
        self.recording_thread.start()

    def _on_stop_record_click(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.status_var.set("Stopping recording...")
        self.btn_stop_rec.config(state=tk.DISABLED)

    def _on_play_sample_click(self):
        if self.recorded_audio is None:
            self.status_var.set("No sample recorded yet")
            return
        self.status_var.set("Playing sample...")
        self.wave_animator.set_state("playback")

        def play_worker():
            try:
                sd.play(self.recorded_audio, self.sample_rate)
                sd.wait()
                self.after(0, lambda: self.status_var.set("Playback complete"))
                self.after(0, lambda: self.wave_animator.set_state("idle"))
            except Exception as e:
                self.after(0, lambda: self.status_var.set(f"Playback error: {str(e)}"))
                self.after(0, lambda: self.wave_animator.set_state("idle"))

        threading.Thread(target=play_worker, daemon=True).start()

    def _on_rerecord_click(self):
        self.recorded_audio = None
        self.recording_data = []
        self.status_var.set("Ready to record")
        self.wave_animator.set_state("idle")
        self.progress["value"] = 0
        self.btn_record.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.DISABLED)

    def _on_save_sample_click(self):
        if self.recorded_audio is None:
            messagebox.showwarning("No Recording", "Please record audio before saving.")
            return
        try:
            # Save to samples/ folder with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = SAMPLES_DIR / f"recorded_{timestamp}.wav"
            SAMPLES_DIR.mkdir(exist_ok=True)
            wavfile.write(str(output_path), self.sample_rate, self.recorded_audio)
            self.status_var.set(f"Saved: {output_path.name}")
            messagebox.showinfo("Success", f"Voice sample saved to:\n{output_path}")
        except Exception as e:
            self.status_var.set(f"Save error: {str(e)}")
            messagebox.showerror("Save Error", f"Failed to save sample:\n{str(e)}")


def main():
    app = TTSApp()
    app._apply_dark_titlebar()
    app.mainloop()


if __name__ == "__main__":
    main()
