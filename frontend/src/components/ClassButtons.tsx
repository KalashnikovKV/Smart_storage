import type { LabelClassOption } from "../api/types";

interface ClassButtonsProps {
  classes: LabelClassOption[];
  disabled?: boolean;
  onConfirm: () => void;
  onSelect: (value: string) => void;
}

export function ClassButtons({ classes, disabled, onConfirm, onSelect }: ClassButtonsProps) {
  return (
    <div>
      <button type="button" className="btn-primary" disabled={disabled} onClick={onConfirm}>
        ✓ Подтвердить предсказание
      </button>
      <div className="class-grid">
        {classes.map((option) => (
          <button
            key={option.hotkey}
            type="button"
            className="class-btn"
            disabled={disabled}
            onClick={() => onSelect(option.value)}
          >
            <kbd>{option.hotkey}</kbd>
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}
