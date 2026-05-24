"""Label class options shared with CLI labeling."""

from __future__ import annotations

from src.app.labeling import LABEL_CLASS_BY_KEY
from src.web.schemas.label import LabelClassOption

# Stable display order for UI buttons.
_LABEL_ORDER: list[tuple[str, str]] = [
    ("1", "Mouse"),
    ("2", "Keyboard"),
    ("3", "Charger Adapter"),
    ("4", "Headphones"),
    ("5", "USB-C Cable"),
    ("6", "Flash Drive"),
    ("7", "Colored Object"),
    ("0", "Unknown Object"),
]


def list_label_classes() -> list[LabelClassOption]:
    """Return class buttons for the labeling UI."""
    options: list[LabelClassOption] = []
    for hotkey, label in _LABEL_ORDER:
        value = LABEL_CLASS_BY_KEY.get(hotkey, label)
        options.append(LabelClassOption(hotkey=hotkey, label=label, value=value))
    return options
