from datetime import datetime
from typing import List
from typing import Optional

from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship


class Base(DeclarativeBase):
    pass


class EvaluationStatus(Base):
    __tablename__ = 'evaluation_status'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)

    evaluation_runs: List['EvaluationRun'] = relationship(
        back_populates='status')


class Snapshot(Base):
    __tablename__ = 'snapshots'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]

    evaluation_runs: List['EvaluationRun'] = relationship(
        back_populates='snapshot')


class Experiment(Base):
    __tablename__ = 'experiments'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)

    evaluation_runs: List['EvaluationRun'] = relationship(
        back_populates='experiment')


class Results(Base):
    __tablename__ = 'results'
    __table_args__ = (UniqueConstraint('evaluation_id',
                                       'metric',
                                       name='_eval_metric'), )
    id: Mapped[int] = mapped_column(primary_key=True)
    evaluation_id = mapped_column(ForeignKey('evaluation_runs.id'))
    metric: Mapped[str]
    value: Mapped[float]

    evaluation: Mapped['EvaluationRun'] = relationship(
        back_populates='results')


class EvaluationRun(Base):
    __tablename__ = 'evaluation_runs'
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
    status_id = mapped_column(ForeignKey('evaluation_status.id'))
    snapshot_id = mapped_column(ForeignKey('snapshots.id'))
    experiment_id = mapped_column(ForeignKey('experiments.id'))

    status: Mapped[EvaluationStatus] = relationship(
        back_populates='evaluation_runs')
    snapshot: Mapped[Snapshot] = relationship(back_populates='evaluation_runs')
    experiment: Mapped[Experiment] = relationship(
        back_populates='evaluation_runs')
    results: List[Results] = relationship(back_populates='evaluation')
