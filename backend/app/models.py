import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    String,
    UniqueConstraint,
    func,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class AionModel(Base):
    __tablename__ = "aion_models"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    model_name = Column(String(100), nullable=False)
    description = Column(String)
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())
    user = relationship("User")


class FactorCategory(Base):
    __tablename__ = "factor_categories"
    id = Column(Integer, primary_key=True)
    category_code = Column(String(10), unique=True, nullable=False)
    name = Column(String(50), nullable=False)
    description = Column(String)


class SystemFactor(Base):
    __tablename__ = "system_factors"
    id = Column(Integer, primary_key=True, autoincrement=True)
    category_id = Column(Integer, ForeignKey("factor_categories.id"), nullable=False)
    factor_code = Column(String(64), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(String)


class ModelFactorWeight(Base):
    __tablename__ = "model_factor_weights"
    model_id = Column(UUID(as_uuid=True), ForeignKey("aion_models.id"), primary_key=True)
    factor_id = Column(Integer, ForeignKey("system_factors.id"), primary_key=True)
    weight = Column(Numeric(5, 2), nullable=False)
    __table_args__ = (PrimaryKeyConstraint("model_id", "factor_id"),)


class ModelCategoryWeight(Base):
    __tablename__ = "model_category_weights"
    model_id = Column(UUID(as_uuid=True), ForeignKey("aion_models.id"), primary_key=True)
    category_id = Column(Integer, ForeignKey("factor_categories.id"), primary_key=True)
    weight = Column(Numeric(5, 2), nullable=False)
    __table_args__ = (PrimaryKeyConstraint("model_id", "category_id"),)


class MarketDataDaily(Base):
    __tablename__ = "market_data_daily"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    # ... other fields as needed


class AnalysisScore(Base):
    __tablename__ = "analysis_scores"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(String(36), unique=True, nullable=False, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    total_score = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=False)
    action_card = Column(String(50))
    factors = Column(JSONB, nullable=False)
    weight_denominator = Column(Float)
    scenarios = Column(JSONB)
    key_variables = Column(JSONB)
    stock_strategy = Column(JSONB)
    option_strategies = Column(JSONB)
    risk_management = Column(JSONB)
    execution_notes = Column(JSONB)
    calculated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, server_default=func.now())
    __table_args__ = (
        Index("ix_analysis_scores_ticker_calculated_at", "ticker", "calculated_at"),
    )


class SoftFactorScore(Base):
    __tablename__ = "soft_factor_scores"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(10), nullable=False, index=True)
    factor_code = Column(String(20), nullable=False)
    asof_date = Column(Date, nullable=False)
    score = Column(Float)
    confidence = Column(Float)
    reasons = Column(JSONB, nullable=False, default=list)
    citations = Column(JSONB, nullable=False, default=list)
    created_at = Column(DateTime, server_default=func.now())
    __table_args__ = (
        UniqueConstraint("ticker", "factor_code", "asof_date", name="uq_soft_factor_scores_identity"),
    )


class ContextStore(Base):
    __tablename__ = "context_store"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(10), nullable=False, index=True)
    asof_date = Column(Date, nullable=False)
    type = Column(String(20), nullable=False)
    summary = Column(JSONB, nullable=False)
    source_refs = Column(JSONB, nullable=False, default=list)
    created_at = Column(DateTime, server_default=func.now())
    __table_args__ = (
        UniqueConstraint("ticker", "asof_date", "type", name="uq_context_store_identity"),
    )


class NarrativeJob(Base):
    __tablename__ = "narrative_jobs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(10), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="pending")
    progress = Column(Integer, nullable=False, default=0)
    error_message = Column(String)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class NarrativeReport(Base):
    __tablename__ = "narrative_reports"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("narrative_jobs.id", ondelete="CASCADE"), nullable=False, unique=True)
    ticker = Column(String(10), nullable=False, index=True)
    output_json = Column(JSONB, nullable=False)
    latency_ms = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())

    job = relationship("NarrativeJob")
