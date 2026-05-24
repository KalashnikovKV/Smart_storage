import { useEffect, useState } from "react";
import { fetchDashboardStats } from "../api/client";
import type { DashboardStats } from "../api/types";

const CHECKLIST_LABELS: Record<string, string> = {
  has_images: "Есть фото в training/test_images",
  has_labels: "Есть ground_truth в CSV",
  has_dataset: "Собран dataset/",
  min_targets_met: "Достигнуты минимумы по классам",
};

export function DashboardPage() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDashboardStats()
      .then(setStats)
      .catch((err) => setError(err instanceof Error ? err.message : "Error"));
  }, []);

  if (error) {
    return <div className="error-banner">{error}</div>;
  }

  if (!stats) {
    return <p className="meta">Загрузка статистики…</p>;
  }

  return (
    <div>
      <h2 className="page-title">Обзор</h2>

      <div className="card" style={{ marginBottom: "1rem" }}>
        <p>
          Изображений: <strong>{stats.total_images}</strong> · размечено:{" "}
          <strong>{stats.labeled_images}</strong> · строк CSV: {stats.csv_rows}
        </p>
        <p className="meta">
          Датасет: train {stats.dataset_train} · val {stats.dataset_val}
        </p>
      </div>

      <div className="card" style={{ marginBottom: "1rem" }}>
        <h3 style={{ marginTop: 0 }}>Чеклист</h3>
        <ul className="checklist">
          {Object.entries(stats.checklist).map(([key, done]) => (
            <li key={key} className={done ? "done" : "pending"}>
              {CHECKLIST_LABELS[key] ?? key}
            </li>
          ))}
        </ul>
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Прогресс по классам</h3>
        <div className="progress-list">
          {stats.class_progress.map((row) => {
            const pct = Math.min(100, Math.round((row.have / row.need) * 100));
            return (
              <div key={row.slug} className="progress-row">
                <span>{row.slug}</span>
                <div className="progress-bar">
                  <div
                    className={`progress-fill ${row.ok ? "ok" : ""}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="meta">
                  {row.have}/{row.need}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
