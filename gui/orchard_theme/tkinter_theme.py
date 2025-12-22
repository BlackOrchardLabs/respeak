import math
import tkinter as tk
from tkinter import ttk

from .colors import (
    BG_MAIN_TOP, BG_MAIN_BOTTOM,
    BG_PANEL, BG_PANEL_ALT, BG_INPUT,
    COBALT, COBALT_SOFT, COBALT_DEEP,
    CYAN_PEAK, CYAN_HALO,
    MAGENTA,
    GRID_SOFT, GRID_HUD,
    TEXT_PRIMARY, TEXT_SUBTLE, TEXT_MUTED,
    BTN_RECORD_IDLE, BTN_RECORD_HOVER, BTN_RECORD_ACTIVE,
    BTN_STOP_IDLE, BTN_STOP_HOVER, BTN_STOP_ACTIVE,
    BTN_PLAY_IDLE, BTN_PLAY_HOVER, BTN_PLAY_ACTIVE,
    BTN_SECONDARY_IDLE, BTN_SECONDARY_HOVER, BTN_SECONDARY_ACTIVE,
    BTN_SPEAK_IDLE, BTN_SPEAK_HOVER, BTN_SPEAK_ACTIVE,
)


BUTTON_ROLE_COLORS = {
    "record": {
        "idle":   BTN_RECORD_IDLE,
        "hover":  BTN_RECORD_HOVER,
        "active": BTN_RECORD_ACTIVE,
        "fg":     TEXT_PRIMARY,
    },
    "stop": {
        "idle":   BTN_STOP_IDLE,
        "hover":  BTN_STOP_HOVER,
        "active": BTN_STOP_ACTIVE,
        "fg":     TEXT_PRIMARY,
    },
    "play": {
        "idle":   BTN_PLAY_IDLE,
        "hover":  BTN_PLAY_HOVER,
        "active": BTN_PLAY_ACTIVE,
        "fg":     TEXT_PRIMARY,
    },
    "speak": {
        "idle":   BTN_SPEAK_IDLE,
        "hover":  BTN_SPEAK_HOVER,
        "active": BTN_SPEAK_ACTIVE,
        "fg":     TEXT_PRIMARY,
    },
    "secondary": {
        "idle":   BTN_SECONDARY_IDLE,
        "hover":  BTN_SECONDARY_HOVER,
        "active": BTN_SECONDARY_ACTIVE,
        "fg":     TEXT_PRIMARY,
    },
}


class OrchardButton(tk.Button):
    def __init__(self, master, role="secondary", *args, **kwargs):
        role_colors = BUTTON_ROLE_COLORS.get(role, BUTTON_ROLE_COLORS["secondary"])

        base_kwargs = {
            "bg": role_colors["idle"],
            "activebackground": role_colors["active"],
            "fg": role_colors["fg"],
            "activeforeground": role_colors["fg"],
            "relief": tk.FLAT,
            "bd": 0,
            "font": ("Segoe UI", 10, "bold"),
            "cursor": "hand2",
            "highlightthickness": 2,
            "highlightbackground": "#000000",
            "highlightcolor": "#000000",
            "padx": 16,
            "pady": 6,
        }
        base_kwargs.update(kwargs)

        super().__init__(master, *args, **base_kwargs)

        self.role_colors = role_colors
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _apply_glow(self, color: str):
        self.config(highlightbackground=color, highlightcolor=color)

    def _on_enter(self, _event):
        self.config(bg=self.role_colors["hover"])
        self._apply_glow(self.role_colors["hover"])

    def _on_leave(self, _event):
        self.config(bg=self.role_colors["idle"])
        self._apply_glow("#000000")

    def _on_press(self, _event):
        self.config(bg=self.role_colors["active"])
        self._apply_glow(self.role_colors["active"])

    def _on_release(self, _event):
        if self.winfo_containing(self.winfo_pointerx(), self.winfo_pointery()) is self:
            self.config(bg=self.role_colors["hover"])
            self._apply_glow(self.role_colors["hover"])
        else:
            self.config(bg=self.role_colors["idle"])
            self._apply_glow("#000000")


def _apply_gradient_background(root: tk.Tk):
    root.update_idletasks()
    w = root.winfo_width() or 600
    h = root.winfo_height() or 550

    canvas = tk.Canvas(root, width=w, height=h, bg=BG_MAIN_TOP, highlightthickness=0, bd=0)
    canvas.place(x=0, y=0, relwidth=1, relheight=1)

    r1, g1, b1 = int(BG_MAIN_TOP[1:3], 16), int(BG_MAIN_TOP[3:5], 16), int(BG_MAIN_TOP[5:7], 16)
    r2, g2, b2 = int(BG_MAIN_BOTTOM[1:3], 16), int(BG_MAIN_BOTTOM[3:5], 16), int(BG_MAIN_BOTTOM[5:7], 16)

    steps = max(h, 1)
    for i in range(steps):
        t = i / steps
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        color = f"#{r:02X}{g:02X}{b:02X}"
        canvas.create_line(0, i, w, i, fill=color)

    # Draw border on the same canvas
    margin = 4
    canvas.create_rectangle(
        margin, margin, w - margin, h - margin,
        outline=GRID_HUD,
        width=1,
    )

    # canvas.lower()
    return canvas


def apply_orchard_window_frame(root: tk.Tk):
    # Set root window background to dark
    root.configure(bg=BG_MAIN_TOP)
    _apply_gradient_background(root)


def style_progressbar(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(
        "Orchard.Horizontal.TProgressbar",
        troughcolor=BG_PANEL,
        background=COBALT,
        bordercolor=BG_PANEL,
        lightcolor=CYAN_HALO,
        darkcolor=COBALT_DEEP,
        thickness=6,
    )


def create_progressbar(parent):
    pb = ttk.Progressbar(
        parent,
        orient="horizontal",
        mode="determinate",
        style="Orchard.Horizontal.TProgressbar",
        length=400,
    )
    pb["value"] = 0
    pb["maximum"] = 100
    return pb


def create_waveform_canvas(parent, width=540, height=180):
    canvas = tk.Canvas(
        parent,
        width=width,
        height=height,
        bg=BG_PANEL_ALT,
        highlightthickness=0,
        bd=0,
    )
    draw_waveform_background(canvas, width, height)
    return canvas


def draw_waveform_background(canvas, width, height):
    canvas.delete("background")

    cx = width // 2
    cy = height // 2

    canvas.create_rectangle(
        0, 0, width, height,
        fill=BG_PANEL_ALT,
        outline="",
        tags="background",
    )

    for x in range(0, width, 40):
        canvas.create_line(x, 0, x, height, fill=GRID_SOFT, width=1, tags="background")
    for y in range(0, height, 40):
        canvas.create_line(0, y, width, y, fill=GRID_SOFT, width=1, tags="background")

    max_radius = min(width, height) // 2 - 20
    step = max(max_radius // 5, 1)
    for r in range(step, max_radius + 1, max(step,1)):
        canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline=COBALT_SOFT,
            width=1,
            tags="background",
        )

    halo = max_radius + 4
    canvas.create_oval(
        cx - halo, cy - halo, cx + halo, cy + halo,
        outline=CYAN_HALO,
        width=1,
        tags="background",
    )

    canvas.create_line(
        cx, 10, cx, height - 10,
        fill=GRID_HUD, width=1, dash=(2, 4), tags="background",
    )
    canvas.create_line(
        10, cy, width - 10, cy,
        fill=GRID_HUD, width=1, dash=(2, 4), tags="background",
    )


class WaveformAnimator:
    def __init__(self, canvas: tk.Canvas, width: int, height: int):
        self.canvas = canvas
        self.width = width
        self.height = height
        self.cx = width // 2
        self.cy = height // 2
        self.phase = 0.0
        self.state = "idle"
        self.line_id = None
        self.running = False
        self.use_external_amplitude = False  # Flag for real-time audio control
        self.current_amplitude = None  # Store the latest amplitude from audio

    def set_state(self, state: str):
        self.state = state
        # When recording, use external amplitude control
        self.use_external_amplitude = (state == "recording")
        if not self.use_external_amplitude:
            self.current_amplitude = None

    def start(self):
        if not self.running:
            self.running = True
            self._tick()

    def stop(self):
        self.running = False

    def _get_color_for_state(self):
        if self.state == "recording":
            return MAGENTA
        elif self.state == "playback":
            return COBALT
        else:
            return GRID_SOFT

    def _tick(self):
        if not self.running:
            return

        self.phase += 0.12

        # Always redraw, but use current_amplitude if in recording mode
        if self.use_external_amplitude and self.current_amplitude is not None:
            self._draw_wave(external_amplitude=self.current_amplitude)
        else:
            self._draw_wave()

        self.canvas.after(30, self._tick)

    def _draw_wave(self, external_amplitude: float = None):
        width = self.width
        height = self.height
        cy = self.cy
        color = self._get_color_for_state()

        if self.line_id is not None:
            self.canvas.delete(self.line_id)

        points = []
        num_samples = 120
        max_amp = height * 0.30

        for i in range(num_samples):
            x = int(width * i / (num_samples - 1))

            if external_amplitude is None:
                amp = (math.sin(self.phase * 0.7) * 0.5 + 0.5)
            else:
                amp = max(0.0, min(1.0, external_amplitude))

            local = math.sin(self.phase + (i * 0.25))
            y = cy - local * amp * max_amp
            points.extend((x, y))

        self.line_id = self.canvas.create_line(
            *points,
            fill=color,
            width=2,
            smooth=True,
        )

    def update_with_amplitude(self, amplitude: float):
        """Update the current amplitude from external source (e.g., microphone)"""
        self.current_amplitude = amplitude
        # The _tick method will handle drawing with this amplitude
