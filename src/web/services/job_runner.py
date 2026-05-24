"""Background subprocess jobs for dataset build and training."""

from __future__ import annotations

import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class JobRecord:
    """One background CLI job."""

    job_id: str
    command: list[str]
    cwd: Path
    status: str = "pending"
    output: str = ""
    exit_code: int | None = None
    _thread: threading.Thread | None = field(default=None, repr=False)


class JobStore:
    """In-memory store for long-running CLI tasks."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def start(self, command: list[str], *, cwd: Path) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        record = JobRecord(job_id=job_id, command=command, cwd=cwd)
        thread = threading.Thread(target=self._run, args=(record,), daemon=True)
        record._thread = thread

        with self._lock:
            self._jobs[job_id] = record

        thread.start()
        return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _run(self, record: JobRecord) -> None:
        record.status = "running"
        try:
            completed = subprocess.run(
                record.command,
                cwd=str(record.cwd),
                capture_output=True,
                text=True,
                check=False,
            )
            record.output = (completed.stdout or "") + (completed.stderr or "")
            record.exit_code = completed.returncode
            record.status = "done" if completed.returncode == 0 else "failed"
        except OSError as exc:
            record.output = str(exc)
            record.exit_code = -1
            record.status = "failed"
