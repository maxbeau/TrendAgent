import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db import AsyncSessionLocal, get_db
from app.models import NarrativeJob, NarrativeReport
from app.services.narrative_orchestrator import generate_narrative

router = APIRouter(prefix="/narrative", tags=["narrative"])
logger = logging.getLogger(__name__)


class NarrativeRequest(BaseModel):
    ticker: constr(strip_whitespace=True, min_length=1, max_length=10, regex=r"^[A-Za-z0-9\.\-]+$")
    model_version: Optional[str] = None


class NarrativeTaskResponse(BaseModel):
    message: str
    task_id: str


async def run_narrative_job(task_id: str, ticker: str, model_version: Optional[str]) -> None:
    job_id = uuid.UUID(task_id)
    async with AsyncSessionLocal() as db:
        job = await db.get(NarrativeJob, job_id)
        if job:
            job.status = "running"
            job.progress = 10
            await db.commit()

        try:
            result = await generate_narrative(ticker, db=db, model_version=model_version)
            report_entry = NarrativeReport(
                job_id=job_id,
                ticker=ticker,
                output_json=result,
                latency_ms=result.get("latency_ms"),
            )
            db.add(report_entry)
            if job:
                job.status = "completed"
                job.progress = 100
                job.error_message = None
            await db.commit()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Narrative task %s failed", task_id)
            if job:
                job.status = "failed"
                job.progress = 100
                job.error_message = str(exc)
                await db.commit()


@router.post("/generate", status_code=202, response_model=NarrativeTaskResponse)
async def generate_narrative_job(
    background_tasks: BackgroundTasks,
    payload: NarrativeRequest,
    db: AsyncSession = Depends(get_db),
) -> NarrativeTaskResponse:
    task_id = str(uuid.uuid4())
    job = NarrativeJob(id=uuid.UUID(task_id), ticker=payload.ticker, status="pending", progress=0)
    db.add(job)
    await db.commit()

    background_tasks.add_task(run_narrative_job, task_id, payload.ticker, payload.model_version)
    return NarrativeTaskResponse(message="Narrative generation started", task_id=task_id)


@router.get("/status/{task_id}")
async def get_narrative_status(task_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    try:
        job_id = uuid.UUID(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid task_id") from exc

    job = await db.get(NarrativeJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")

    stmt = select(NarrativeReport).where(NarrativeReport.job_id == job_id)
    result = await db.execute(stmt)
    report = result.scalars().first()

    if job.status == "completed" and report:
        return {
            "status": "completed",
            "progress": job.progress,
            "data": report.output_json,
        }

    if job.status == "failed":
        return {
            "status": "failed",
            "progress": job.progress,
            "error": job.error_message,
        }

    return {
        "status": job.status,
        "progress": job.progress,
    }
