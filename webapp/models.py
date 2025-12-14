from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Float, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class TickerList(Base):
    __tablename__ = "ticker_lists"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    is_watch_list = Column(Boolean, default=False)

    tickers = relationship("Ticker", back_populates="ticker_list", cascade="all, delete-orphan")

class Ticker(Base):
    __tablename__ = "tickers"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    list_id = Column(Integer, ForeignKey("ticker_lists.id"))

    ticker_list = relationship("TickerList", back_populates="tickers")


class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    synthetic_symbol = Column(String, unique=True, index=True, nullable=False)
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    positions = relationship(
        "PortfolioPosition",
        back_populates="portfolio",
        cascade="all, delete-orphan",
    )
    trading_logs = relationship(
        "TradingLogEntry",
        back_populates="portfolio",
        cascade="all, delete-orphan",
    )
    broker_connectors = relationship(
        "BrokerConnector",
        back_populates="portfolio",
        cascade="all, delete-orphan",
    )
    simulation_groups = relationship(
        "SimulationGroup",
        back_populates="portfolio",
        cascade="all, delete-orphan",
    )


class PortfolioPosition(Base):
    __tablename__ = "portfolio_positions"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False, index=True)
    ticker = Column(String, index=True, nullable=False)
    quantity = Column(Float, default=0.0)
    avg_cost = Column(Float, default=0.0)
    current_price = Column(Float, default=0.0)
    sector = Column(String, default="")
    weight = Column(Float, default=0.0)
    notes = Column(Text, default="")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    portfolio = relationship("Portfolio", back_populates="positions")


class TradingLogEntry(Base):
    __tablename__ = "trading_log_entries"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False, index=True)
    executed_at = Column(DateTime, default=datetime.utcnow, index=True)
    action = Column(String, nullable=False)  # buy/sell
    ticker = Column(String, nullable=False, index=True)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    source = Column(String, default="manual")
    raw_payload = Column(Text, default="")

    # Journal metadata for AI analysis and user recall
    strategy_name = Column(String, default="")
    setup_name = Column(String, default="")
    thesis = Column(Text, default="")
    entry_plan = Column(Text, default="")
    exit_plan = Column(Text, default="")
    risk_notes = Column(Text, default="")
    risk_amount = Column(Float)
    confidence = Column(Float)
    emotion = Column(String, default="")
    journal_notes = Column(Text, default="")
    tags = Column(String, default="")

    # Simulation + attribution fields for multi-user, multi-strategy usage
    is_simulated = Column(Boolean, default=False, index=True)
    simulation_group = Column(String, default="")
    created_by = Column(String, default="system", index=True)
    strategy_source = Column(String, default="")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    portfolio = relationship("Portfolio", back_populates="trading_logs")


class SimulationGroup(Base):
    __tablename__ = "simulation_groups"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    slug = Column(String, nullable=False, index=True)
    hypothesis = Column(Text, default="")
    status = Column(String, default="active", index=True)
    guardrail_max_positions = Column(Integer)
    guardrail_max_capital = Column(Float)
    follow_up_date = Column(DateTime)
    auto_tasks = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    portfolio = relationship("Portfolio", back_populates="simulation_groups")


class BrokerConnector(Base):
    __tablename__ = "broker_connectors"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    credentials = Column(Text, default="")
    last_status = Column(String, default="never_synced")
    last_message = Column(Text, default="")
    last_synced_at = Column(DateTime)

    portfolio = relationship("Portfolio", back_populates="broker_connectors")
