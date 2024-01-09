from datetime import datetime
from typing import List
from typing import Optional

from sqlalchemy import UniqueConstraint, Index, Column, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Session, Mapped
from sqlalchemy.orm import mapped_column, relationship


class Base(DeclarativeBase):
    pass


class EvaluationStatus(Base):
    __tablename__ = 'evaluation_status'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)

    evaluation_runs: Mapped[List['EvaluationRun']] = relationship(
        back_populates='status')


class Experiment(Base):
    __tablename__ = 'experiments'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)

    evaluation_runs: Mapped[List['EvaluationRun']] = relationship(
        back_populates='experiment')


class Metric(Base):
    __tablename__ = 'metrics'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)


class Results(Base):
    __tablename__ = 'results'
    __table_args__ = (UniqueConstraint('evaluation_id',
                                       'metric_id',
                                       name='_eval_metric'), )
    id: Mapped[int] = mapped_column(primary_key=True)
    evaluation_id = mapped_column(ForeignKey('evaluation_runs.id'))
    metric_id = mapped_column(ForeignKey('metrics.id'))
    value: Mapped[float] = mapped_column(nullable=True)


class EvaluationRun(Base):
    __tablename__ = 'evaluation_runs'
    __table_args__ = (UniqueConstraint('experiment_id',
                                       'snapshot',
                                       name='_exp_snapshot'), )

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False,
                                                 default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(nullable=True,
                                                 onupdate=datetime.now)
    status_id = mapped_column(ForeignKey('evaluation_status.id'),
                              nullable=False)
    # snapshot_id = mapped_column(ForeignKey('snapshots.id'), nullable=False)
    experiment_id = mapped_column(ForeignKey('experiments.id'), nullable=False)

    snapshot: Mapped[str] = mapped_column(nullable=False)

    status: Mapped[EvaluationStatus] = relationship(
        back_populates='evaluation_runs')
    # snapshot: Mapped[Snapshot] = relationship(back_populates='evaluation_runs')
    experiment: Mapped[Experiment] = relationship(
        back_populates='evaluation_runs')


def get_or_create(engine, model, **kwargs):
    with Session(engine) as session, session.begin():
        instance = session.query(model).filter_by(**kwargs).first()
        if instance:
            return instance
        else:
            instance = model(**kwargs)
            session.add(instance)
            return instance


def create_tables(engine):
    Base.metadata.create_all(engine)
