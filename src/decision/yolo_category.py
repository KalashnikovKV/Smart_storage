"""Map raw YOLO class names to canonical app category strings."""

_YOLO_CATEGORY_MAP: dict[str, str] = {
    "mouse": "Mouse",
    "keyboard": "Keyboard",
    "charger": "Charger Adapter",
    "charger_adapter": "Charger Adapter",
    "cable": "USB-C Cable",
    "usb_cable": "USB-C Cable",
    "usb-c_cable": "USB-C Cable",
    "headphones": "Headphones",
    "flash_drive": "Flash Drive",
    "flashdrive": "Flash Drive",
    "colored_object": "Colored Object",
}


def yolo_category(raw: str) -> str:
    return _YOLO_CATEGORY_MAP.get(raw.lower().replace(" ", "_"), raw.title())
