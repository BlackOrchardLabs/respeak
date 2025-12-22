#!/usr/bin/env python3
"""
TTS GUI - Simple Text-to-Speech Interface
Supports Voice Kernel (custom voice) or system TTS fallback
With kernel selector dropdown.
"""

import os
import sys
import json
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
from pathlib import Path
import threading
import tempfile

# Add repo root to path
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
                wav_path = REPO_ROOT / speaker_wav
                if wav_path.exists():
                    return wav_path
                wav_path = SAMPLES_DIR / Path(speaker_wav).name
                if wav_path.exists():
                    return wav_path
    except Exception as e:
        print(f"Warning: Could not read speaker_wav from {kernel_path}: {e}")
    return None


class TTSGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Re:speak TTS")
        self.root.geometry("620x580")

        # TTS state
        self.tts_mode = "system"
        self.tts_engine = None
        self.voice_kernel = None
        self.counter = 1
        self.is_speaking = False

        # Kernel selection
        self.available_kernels = scan_kernels()
        self.selected_kernel_path = None
        self.selected_sample_path = None

        # Configure window
        self.root.configure(bg="#1e1e1e")

        # Title label
        title_label = tk.Label(
            root, text="Re:speak TTS",
            font=("Arial", 18, "bold"), bg="#1e1e1e", fg="#ffffff"
        )
        title_label.pack(pady=10)

        # Kernel selector frame
        kernel_frame = tk.Frame(root, bg="#1e1e1e")
        kernel_frame.pack(pady=5)

        tk.Label(
            kernel_frame, text="Voice Kernel:",
            font=("Arial", 10), bg="#1e1e1e", fg="#ffffff"
        ).pack(side=tk.LEFT, padx=5)

        if self.available_kernels:
            kernel_names = [name for name, path in self.available_kernels]
        else:
            kernel_names = ["(no kernels found)"]

        self.kernel_var = tk.StringVar(value=kernel_names[0] if kernel_names else "")
        self.kernel_dropdown = ttk.Combobox(
            kernel_frame, textvariable=self.kernel_var,
            values=kernel_names, width=30, state="readonly"
        )
        self.kernel_dropdown.pack(side=tk.LEFT, padx=5)
        self.kernel_dropdown.bind("<<ComboboxSelected>>", self._on_kernel_selected)

        refresh_btn = tk.Button(
            kernel_frame, text="Refresh", command=self._refresh_kernels,
            bg="#4a4a4a", fg="#ffffff", padx=10
        )
        refresh_btn.pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_label = tk.Label(
            root, text="Initializing...",
            font=("Arial", 9), bg="#1e1e1e", fg="#888888"
        )
        self.status_label.pack(pady=2)

        # Word counter label
        self.word_count_label = tk.Label(
            root, text="Words: 0 / 1000",
            font=("Arial", 10), bg="#1e1e1e", fg="#888888"
        )
        self.word_count_label.pack(pady=5)

        # Text input area
        self.text_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, width=70, height=18,
            font=("Arial", 11), bg="#2d2d2d", fg="#ffffff",
            insertbackground="#ffffff", selectbackground="#4a4a4a"
        )
        self.text_area.pack(padx=20, pady=10)
        self.text_area.bind("<KeyRelease>", self.update_word_count)

        # Button frame
        button_frame = tk.Frame(root, bg="#1e1e1e")
        button_frame.pack(pady=10)

        self.speak_button = tk.Button(
            button_frame, text="Speak", command=self.speak_text,
            font=("Arial", 12, "bold"), bg="#0066cc", fg="#ffffff",
            activebackground="#0052a3", activeforeground="#ffffff",
            padx=30, pady=10, cursor="hand2"
        )
        self.speak_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(
            button_frame, text="Stop", command=self.stop_speaking,
            font=("Arial", 12, "bold"), bg="#cc0000", fg="#ffffff",
            activebackground="#a30000", activeforeground="#ffffff",
            padx=30, pady=10, cursor="hand2", state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(
            button_frame, text="Clear", command=self.clear_text,
            font=("Arial", 12), bg="#4a4a4a", fg="#ffffff",
            activebackground="#363636", activeforeground="#ffffff",
            padx=20, pady=10, cursor="hand2"
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # Initialize TTS
        self._initialize_tts()

    def _refresh_kernels(self):
        """Refresh the kernel list from disk"""
        self.available_kernels = scan_kernels()
        if self.available_kernels:
            kernel_names = [name for name, path in self.available_kernels]
            self.kernel_dropdown['values'] = kernel_names
            self.status_label.config(text=f"Found {len(self.available_kernels)} kernel(s)")
        else:
            self.kernel_dropdown['values'] = ["(no kernels found)"]
            self.status_label.config(text="No kernels found")

    def _on_kernel_selected(self, event=None):
        """Handle kernel selection change"""
        selected_name = self.kernel_var.get()
        for name, path in self.available_kernels:
            if name == selected_name:
                self.selected_kernel_path = path
                self.selected_sample_path = get_speaker_wav_from_kernel(path)
                break
        self.status_label.config(text=f"Loading {selected_name}...")
        self.root.update()
        self._initialize_tts()

    def _initialize_tts(self):
        """Initialize TTS engine"""
        if not self.selected_kernel_path and self.available_kernels:
            name, path = self.available_kernels[0]
            self.selected_kernel_path = path
            self.selected_sample_path = get_speaker_wav_from_kernel(path)
            self.kernel_var.set(name)

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

                    def load_audio_patch(audiopath, load_sr=22050):
                        audio, sr = sf.read(audiopath)
                        audio_tensor = torch.FloatTensor(audio)
                        if len(audio_tensor.shape) > 1:
                            audio_tensor = audio_tensor.mean(dim=1)
                        if len(audio_tensor.shape) == 1:
                            audio_tensor = audio_tensor.unsqueeze(0)
                        return audio_tensor

                    import TTS.tts.models.xtts as xtts_module
                    xtts_module.load_audio = load_audio_patch

                    print(f"Loading voice kernel from {self.selected_kernel_path}...")
                    self.voice_kernel = load_kernel(
                        str(self.selected_kernel_path),
                        audio_path=str(self.selected_sample_path)
                    )
                    print(f"Loaded kernel: {self.voice_kernel.name}")

                    if self.tts_engine is None or self.tts_mode != "voice_kernel":
                        print("Initializing Coqui XTTS v2...")
                        self.tts_engine = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
                        print("Voice Kernel TTS ready!")

                    self.tts_mode = "voice_kernel"
                    self.status_label.config(text=f"Voice Kernel: {self.voice_kernel.name}")
                    return

                except Exception as e:
                    print(f"Failed to initialize Voice Kernel: {e}")
                    import traceback
                    traceback.print_exc()

        print("Using system TTS...")
        self._init_system_tts()

    def _init_system_tts(self):
        """Initialize pyttsx3 system TTS"""
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_mode = "system"
            self.status_label.config(text="System TTS (no kernel loaded)")
        except Exception as e:
            print(f"Failed to initialize system TTS: {e}")
            self.tts_mode = "none"
            self.status_label.config(text="No TTS available")

    def count_words(self, text):
        return len(text.split())

    def update_word_count(self, event=None):
        text = self.text_area.get("1.0", tk.END).strip()
        word_count = self.count_words(text)
        color = "#cc0000" if word_count > 1000 else "#888888"
        self.word_count_label.config(fg=color, text=f"Words: {word_count} / 1000")

    def speak_text(self):
        text = self.text_area.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("No Text", "Please enter some text to speak.")
            return

        word_count = self.count_words(text)
        if word_count > 1000:
            messagebox.showerror("Text Too Long", f"Text contains {word_count} words. Maximum is 1000.")
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
        try:
            self.is_speaking = True
            self.speak_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)

            output_file = Path(tempfile.gettempdir()) / f"respeak_tts_{self.counter:03d}.wav"
            self.counter += 1

            print(f"Generating speech with Voice Kernel...")
            self.tts_engine.tts_to_file(
                text=text, file_path=str(output_file),
                speaker_wav=self.voice_kernel.get_speaker_wav(),
                language="en"
            )
            print(f"Playing: {output_file}")
            os.system(f'start "" "{output_file}"')

        except Exception as e:
            messagebox.showerror("TTS Error", f"Error: {str(e)}")
        finally:
            self.is_speaking = False
            self.speak_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def _speak_system(self, text):
        try:
            self.is_speaking = True
            self.speak_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            messagebox.showerror("TTS Error", f"Error: {str(e)}")
        finally:
            self.is_speaking = False
            self.speak_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def stop_speaking(self):
        if self.is_speaking:
            if self.tts_mode == "system":
                self.tts_engine.stop()
            self.is_speaking = False
            self.speak_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def clear_text(self):
        self.text_area.delete("1.0", tk.END)
        self.update_word_count()


def main():
    root = tk.Tk()
    app = TTSGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
