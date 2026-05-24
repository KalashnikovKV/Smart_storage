import type {
  DashboardStats,
  DatasetBuildRequest,
  JobResponse,
  JobStatusResponse,
  LabelFilter,
  LabelNextResponse,
  LabelSubmitResponse,
  TrainRequest,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json() as Promise<T>;
}

export function fetchLabelNext(filter: LabelFilter): Promise<LabelNextResponse> {
  return request(`/api/label/next?filter=${filter}`);
}

export function submitLabel(
  imageName: string,
  options: { confirm_prediction?: boolean; ground_truth?: string; skip?: boolean },
  filter: LabelFilter,
): Promise<LabelSubmitResponse> {
  return request(`/api/label/submit?filter=${filter}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_name: imageName,
      confirm_prediction: options.confirm_prediction ?? false,
      ground_truth: options.ground_truth ?? "",
      skip: options.skip ?? false,
    }),
  });
}

export function fetchDashboardStats(): Promise<DashboardStats> {
  return request("/api/dashboard/stats");
}

export function startDatasetBuild(body: DatasetBuildRequest): Promise<JobResponse> {
  return request("/api/dataset/build", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function fetchDatasetJob(jobId: string): Promise<JobStatusResponse> {
  return request(`/api/dataset/jobs/${jobId}`);
}

export function startTraining(body: TrainRequest): Promise<JobResponse> {
  return request("/api/train/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function fetchTrainJob(jobId: string): Promise<JobStatusResponse> {
  return request(`/api/train/jobs/${jobId}`);
}
