# Black Orchid Holographic Palette - colors.py

# Deep gradient space background
BG_MAIN_TOP       = "#050A16"
BG_MAIN_BOTTOM    = "#0A1224"

# Panels
BG_PANEL          = "#0C1629"
BG_PANEL_ALT      = "#081122"
BG_INPUT          = "#0E1A33"

# Neon blues (true holographic cobalt)
COBALT            = "#00A8FF"
COBALT_SOFT       = "#0090E6"
COBALT_DEEP       = "#006BB8"

# Cyan peaks
CYAN_PEAK         = "#4CFFFF"
CYAN_HALO         = "#00F0FF"

# Magenta accents
MAGENTA           = "#FF3DAF"

# Subtle grid lines
GRID_SOFT         = "#1A2E4A"
GRID_HUD          = "#103055"

# Text
TEXT_PRIMARY      = "#C7DAFF"
TEXT_SUBTLE       = "#8CA3C9"
TEXT_MUTED        = "#4F5B78"

# Buttons (derived from reference lighting)
BTN_RECORD_IDLE   = MAGENTA
BTN_RECORD_HOVER  = "#FF65C2"
BTN_RECORD_ACTIVE = "#FF8AD2"

BTN_STOP_IDLE     = "#CC0022"
BTN_STOP_HOVER    = "#FF335C"
BTN_STOP_ACTIVE   = "#FF6685"

BTN_PLAY_IDLE     = COBALT
BTN_PLAY_HOVER    = CYAN_HALO
BTN_PLAY_ACTIVE   = CYAN_PEAK

BTN_SECONDARY_IDLE   = "#0D233F"
BTN_SECONDARY_HOVER  = COBALT_DEEP
BTN_SECONDARY_ACTIVE = COBALT

BTN_SPEAK_IDLE       = COBALT
BTN_SPEAK_HOVER      = CYAN_HALO
BTN_SPEAK_ACTIVE     = CYAN_PEAK

# --- Backwards compatibility aliases for older UI code ---
BG_MAIN = BG_PANEL
BG_PANEL_BG = BG_PANEL
BG_PANEL_ALT_BG = BG_PANEL_ALT
