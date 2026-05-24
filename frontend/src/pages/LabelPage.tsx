import { useCallback, useEffect, useState } from "react";
import { fetchLabelNext, submitLabel } from "../api/client";
import type { LabelClassOption, LabelFilter, LabelItem, LabelQueueStats } from "../api/types";
import { ClassButtons } from "../components/ClassButtons";

export function LabelPage() {
  const [filter, setFilter] = useState<LabelFilter>("unlabeled");
  const [item, setItem] = useState<LabelItem | null>(null);
  const [classes, setClasses] = useState<LabelClassOption[]>([]);
  const [stats, setStats] = useState<LabelQueueStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadNext = useCallback(async (activeFilter: LabelFilter = filter) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchLabelNext(activeFilter);
      setItem(response.item);
      setClasses(response.classes);
      setStats(response.stats);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load image");
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    void loadNext(filter);
  }, [filter, loadNext]);

  const handleSubmit = async (options: {
    confirm_prediction?: boolean;
    ground_truth?: string;
    skip?: boolean;
  }) => {
    if (!item) return;
    setLoading(true);
    setError(null);
    try {
      const response = await submitLabel(item.image_name, options, filter);
      setItem(response.next_item);
      setStats(response.stats);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save label");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (loading || !item) return;
      if (event.key === "Enter") {
        event.preventDefault();
        void handleSubmit({ confirm_prediction: true });
        return;
      }
      const match = classes.find((option) => option.hotkey === event.key);
      if (match) {
        event.preventDefault();
        void handleSubmit({ ground_truth: match.value });
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [classes, item, loading, filter]);

  const labeledCount = stats ? stats.labeled : 0;
  const totalCount = stats ? stats.total : 0;
  const remaining = stats ? stats.remaining : 0;

  return (
    <div>
      <h2 className="page-title">Разметка</h2>

      <div className="toolbar">
        <select value={filter} onChange={(e) => setFilter(e.target.value as LabelFilter)}>
          <option value="unlabeled">Только без label</option>
          <option value="low_confidence">Низкий confidence</option>
          <option value="all">Все изображения</option>
        </select>
        <span className="stats-pill">
          {labeledCount} / {totalCount} размечено · осталось {remaining}
        </span>
        <button type="button" className="class-btn" disabled={loading} onClick={() => void loadNext()}>
          Обновить
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {!item && !loading && (
        <div className="card empty-state">Нет изображений в очереди. Добавьте фото в training/test_images/</div>
      )}

      {item && (
        <div className="label-layout">
          <div className="card">
            <div className="preview-box">
              <img
                src={`${item.preview_url}?t=${item.image_name}`}
                alt={item.image_name}
              />
            </div>
            <p className="meta" style={{ marginTop: "0.75rem" }}>
              {item.image_name}
              {item.processing_time_ms > 0 && ` · ${Math.round(item.processing_time_ms)} ms`}
            </p>
          </div>

          <div className="card">
            {item.has_detection ? (
              <div className="prediction-card">
                <h3>{item.predicted_category}</h3>
                <p className="meta">
                  {Math.round(item.confidence * 100)}% · {item.color} · {item.size}
                </p>
                <p className="meta">method: {item.method_used}</p>
              </div>
            ) : (
              <div className="prediction-card">
                <h3>Объект не найден</h3>
                <p className="meta">Pipeline не смог детектировать объект на этом кадре.</p>
              </div>
            )}

            <ClassButtons
              classes={classes}
              disabled={loading || !item.has_detection}
              onConfirm={() => void handleSubmit({ confirm_prediction: true })}
              onSelect={(value) => void handleSubmit({ ground_truth: value })}
            />
            {!item.has_detection && (
              <button
                type="button"
                className="class-btn"
                style={{ marginTop: "0.75rem", width: "100%" }}
                disabled={loading}
                onClick={() => void handleSubmit({ skip: true })}
              >
                Пропустить → следующая
              </button>
            )}
          </div>
        </div>
      )}

      {loading && <p className="meta">Загрузка…</p>}
    </div>
  );
}
