import { useEffect, useState } from "react";
import { fetchTrainJob, startTraining } from "../api/client";

export function TrainPage() {
  const [validateOnly, setValidateOnly] = useState(true);
  const [model, setModel] = useState("yolo11n-seg.pt");
  const [epochs, setEpochs] = useState(100);
  const [batch, setBatch] = useState(8);
  const [device, setDevice] = useState("cpu");
  const [jobId, setJobId] = useState<string | null>(null);
  const [log, setLog] = useState("");
  const [status, setStatus] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    const timer = setInterval(() => {
      fetchTrainJob(jobId)
        .then((job) => {
          setLog(job.output);
          setStatus(job.status);
          if (job.status === "done" || job.status === "failed") {
            clearInterval(timer);
          }
        })
        .catch(() => clearInterval(timer));
    }, 2000);
    return () => clearInterval(timer);
  }, [jobId]);

  const handleStart = async () => {
    setLog("");
    setStatus("starting");
    const job = await startTraining({
      validate_only: validateOnly,
      model,
      epochs,
      imgsz: 640,
      batch,
      device,
    });
    setJobId(job.job_id);
    setStatus(job.status);
  };

  return (
    <div>
      <h2 className="page-title">Обучение YOLO-Seg</h2>

      <div className="card form-grid">
        <label>
          <input
            type="checkbox"
            checked={validateOnly}
            onChange={(e) => setValidateOnly(e.target.checked)}
          />
          --validate-only (только проверка датасета)
        </label>
        <label>
          model
          <input type="text" value={model} onChange={(e) => setModel(e.target.value)} />
        </label>
        <label>
          epochs
          <input type="number" min={1} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
        </label>
        <label>
          batch
          <input type="number" min={1} value={batch} onChange={(e) => setBatch(Number(e.target.value))} />
        </label>
        <label>
          device
          <select value={device} onChange={(e) => setDevice(e.target.value)}>
            <option value="cpu">cpu</option>
            <option value="cuda">cuda</option>
            <option value="mps">mps</option>
          </select>
        </label>
        <button type="button" className="btn-primary" onClick={() => void handleStart()}>
          {validateOnly ? "Validate dataset" : "Start training"}
        </button>
      </div>

      {status && <p className="meta">Статус: {status}</p>}
      {log && <pre className="log-box">{log}</pre>}
    </div>
  );
}
