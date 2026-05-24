export interface LabelClassOption {
  hotkey: string;
  label: string;
  value: string;
}

export interface LabelItem {
  image_name: string;
  image_url: string;
  preview_url: string;
  predicted_category: string;
  confidence: number;
  color: string;
  size: string;
  method_used: string;
  processing_time_ms: number;
  has_detection: boolean;
}

export interface LabelQueueStats {
  total: number;
  labeled: number;
  remaining: number;
  filter: string;
}

export interface LabelNextResponse {
  item: LabelItem | null;
  stats: LabelQueueStats;
  classes: LabelClassOption[];
}

export interface LabelSubmitResponse {
  saved_ground_truth: string;
  next_item: LabelItem | null;
  stats: LabelQueueStats;
}

export interface ClassProgress {
  slug: string;
  have: number;
  need: number;
  ok: boolean;
}

export interface DashboardStats {
  total_images: number;
  labeled_images: number;
  csv_rows: number;
  dataset_train: number;
  dataset_val: number;
  class_progress: ClassProgress[];
  checklist: Record<string, boolean>;
}

export interface JobResponse {
  job_id: string;
  status: string;
  command: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: string;
  command: string;
  output: string;
  exit_code: number | null;
}

export type LabelFilter = "unlabeled" | "low_confidence" | "all";

export interface DatasetBuildRequest {
  clean: boolean;
  require_ground_truth: boolean;
  regenerate_masks: boolean;
  min_confidence: number;
  val_ratio: number;
}

export interface TrainRequest {
  validate_only: boolean;
  model: string;
  epochs: number;
  imgsz: number;
  batch: number;
  device: string;
}
