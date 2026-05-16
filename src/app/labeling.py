"""Interactive ground-truth labeling helpers for dataset collection."""

from __future__ import annotations

import threading
from collections.abc import Callable

# Menu key → canonical category stored in CSV ground_truth column
LABEL_CLASS_BY_KEY: dict[str, str] = {
    "1": "Mouse",
    "2": "Keyboard",
    "3": "Charger Adapter",
    "4": "Headphones",
    "5": "USB-C Cable",
    "6": "Flash Drive",
    "7": "Colored Object",
    "0": "Unknown Object",
}

LABEL_MENU_LINES = (
    "  1=Mouse  2=Keyboard  3=Charger  4=Headphones",
    "  5=Cable  6=FlashDrive  7=ColoredObject  0=Unknown",
)


def read_choice_while_pumping(
    prompt: str,
    input_reader: Callable[[str], str],
    pump: Callable[[], None],
) -> str:
    """Read terminal input in a thread while pump() keeps the OpenCV window responsive."""
    holder: list[str] = []

    def worker() -> None:
        try:
            holder.append(input_reader(prompt).strip())
        except EOFError:
            holder.append("")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while thread.is_alive():
        pump()
    thread.join()
    return holder[0] if holder else ""


def resolve_ground_truth(predicted_category: str, choice: str) -> str:
    """Map user input to ground_truth; empty string confirms the pipeline prediction."""
    if choice == "":
        return predicted_category
    return LABEL_CLASS_BY_KEY.get(choice, predicted_category)


def prompt_ground_truth(
    predicted_category: str,
    confidence: float,
    input_fn=input,
    pump: Callable[[], None] | None = None,
) -> str:
    """Ask the user to confirm or override the pipeline category."""
    print(f"\n[Pipeline предсказал: {predicted_category} — {confidence:.0%}]")
    print("\nПодтвердить? Enter=да  или введи класс:")
    for line in LABEL_MENU_LINES:
        print(line)

    try:
        if pump is None:
            choice = input_fn("> ").strip()
        else:
            choice = read_choice_while_pumping("> ", input_fn, pump)
    except EOFError:
        return predicted_category

    return resolve_ground_truth(predicted_category, choice)
