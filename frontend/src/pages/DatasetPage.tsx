import { useEffect, useState } from "react";
import { fetchDashboardStats, fetchDatasetJob, startDatasetBuild } from "../api/client";

export function DatasetPage() {
  const [clean, setClean] = useState(true);
  const [requireGroundTruth, setRequireGroundTruth] = useState(false);
  const [regenerateMasks, setRegenerateMasks] = useState(false);
  const [minConfidence, setMinConfidence] = useState(0);
  const [valRatio, setValRatio] = useState(0.2);
  const [jobId, setJobId] = useState<string | null>(null);
  const [log, setLog] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [datasetInfo, setDatasetInfo] = useState("");

  useEffect(() => {
    fetchDashboardStats()
      .then((stats) => {
        setDatasetInfo(`train ${stats.dataset_train} · val ${stats.dataset_val}`);
      })
      .catch(() => undefined);
  }, [status]);

  useEffect(() => {
    if (!jobId) return;
    const timer = setInterval(() => {
      fetchDatasetJob(jobId)
        .then((job) => {
          setLog(job.output);
          setStatus(job.status);
          if (job.status === "done" || job.status === "failed") {
            clearInterval(timer);
          }
        })
        .catch(() => clearInterval(timer));
    }, 1500);
    return () => clearInterval(timer);
  }, [jobId]);

  const handleBuild = async () => {
    setLog("");
    setStatus("starting");
    const job = await startDatasetBuild({
      clean,
      require_ground_truth: requireGroundTruth,
      regenerate_masks: regenerateMasks,
      min_confidence: minConfidence,
      val_ratio: valRatio,
    });
    setJobId(job.job_id);
    setStatus(job.status);
  };

  return (
    <div>
      <h2 className="page-title">Сборка датасета</h2>
      <p className="meta">Текущий dataset: {datasetInfo || "—"}</p>

      <div className="card form-grid">
        <label>
          <input type="checkbox" checked={clean} onChange={(e) => setClean(e.target.checked)} />
          --clean (удалить dataset/ перед сборкой)
        </label>
        <label>
          <input
            type="checkbox"
            checked={requireGroundTruth}
            onChange={(e) => setRequireGroundTruth(e.target.checked)}
          />
          --require-ground-truth
        </label>
        <label>
          <input
            type="checkbox"
            checked={regenerateMasks}
            onChange={(e) => setRegenerateMasks(e.target.checked)}
          />
          --regenerate-masks
        </label>
        <label>
          min-confidence
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={minConfidence}
            onChange={(e) => setMinConfidence(Number(e.target.value))}
          />
        </label>
        <label>
          val-ratio
          <input
            type="number"
            min={0.05}
            max={0.5}
            step={0.05}
            value={valRatio}
            onChange={(e) => setValRatio(Number(e.target.value))}
          />
        </label>
        <button type="button" className="btn-primary" onClick={() => void handleBuild()}>
          Собрать датасет
        </button>
      </div>

      {status && <p className="meta">Статус: {status}</p>}
      {log && <pre className="log-box">{log}</pre>}
    </div>
  );
}
