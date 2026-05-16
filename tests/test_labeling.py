"""Tests for interactive labeling helpers."""

import threading

from src.app.labeling import (
    prompt_ground_truth,
    read_choice_while_pumping,
    resolve_ground_truth,
)


def test_resolve_ground_truth_confirm_prediction():
    assert resolve_ground_truth("Charger Adapter", "") == "Charger Adapter"


def test_resolve_ground_truth_menu_keys():
    assert resolve_ground_truth("Mouse", "3") == "Charger Adapter"
    assert resolve_ground_truth("Mouse", "0") == "Unknown Object"


def test_resolve_ground_truth_invalid_key_keeps_prediction():
    assert resolve_ground_truth("Keyboard", "99") == "Keyboard"


def test_prompt_ground_truth_confirm_with_enter():
    result = prompt_ground_truth("Mouse", 0.84, input_fn=lambda _: "")
    assert result == "Mouse"


def test_prompt_ground_truth_override():
    result = prompt_ground_truth(
        "Charger Adapter",
        0.84,
        input_fn=lambda _: "5",
    )
    assert result == "USB-C Cable"


def test_read_choice_while_pumping_calls_pump_until_input_done():
    pump_calls = []
    input_ready = threading.Event()

    def input_reader(prompt: str) -> str:
        input_ready.wait(timeout=2)
        return "3"

    def pump() -> None:
        pump_calls.append(1)
        input_ready.set()

    choice = read_choice_while_pumping("> ", input_reader, pump)
    assert choice == "3"
    assert len(pump_calls) >= 1


def test_prompt_ground_truth_with_pump():
    pumped = []
    input_ready = threading.Event()

    def input_fn(prompt: str) -> str:
        input_ready.wait(timeout=2)
        return "1"

    def pump() -> None:
        pumped.append(1)
        input_ready.set()

    result = prompt_ground_truth(
        "Mouse",
        0.9,
        input_fn=input_fn,
        pump=pump,
    )
    assert result == "Mouse"
    assert len(pumped) >= 1
