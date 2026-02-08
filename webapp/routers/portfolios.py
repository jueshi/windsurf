from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Tuple
import hashlib
import re
import calendar
import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Body, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, constr
from sqlalchemy import func
from sqlalchemy.orm import Session
import markdown
import yfinance as yf

from ..database import get_db
from .. import models
from .. import gemini_analyzer
from .. import ocr_pipeline
from ..strategy_cache import strategy_cache

router = APIRouter(prefix="/portfolios", tags=["portfolios"])

BROKER_PROVIDERS = {
    "schwab": "Charles Schwab",
    "fidelity": "Fidelity",
    "etrade": "E*TRADE",
    "robinhood": "Robinhood",
    "ibkr": "Interactive Brokers",
    "tda": "TD Ameritrade",
    "other": "Other / Custom",
}

class StartingPosition(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    target_value: float = Field(..., gt=0)


class PortfolioCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    notes: Optional[str] = None
    synthetic_symbol: Optional[str] = Field(None, max_length=20)
    starting_positions: Optional[List[StartingPosition]] = None


class PositionPayload(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    quantity: float = 0.0
    avg_cost: float = 0.0
    current_price: float = 0.0
    sector: Optional[str] = ""
    weight: float = 0.0
    notes: Optional[str] = ""


ActionField = constr(pattern=r"^(?i)(buy|sell)$")


class TradingLogEntryPayload(BaseModel):
    action: ActionField
    ticker: str = Field(..., min_length=1, max_length=10)
    quantity: float = Field(..., gt=0)
    price: float = Field(..., gt=0)
    executed_at: Optional[datetime] = None
    source: Optional[str] = "manual"
    raw_payload: Optional[str] = None

    # Journal metadata
    strategy_name: Optional[str] = Field(None, max_length=120)
    setup_name: Optional[str] = Field(None, max_length=120)
    thesis: Optional[str] = Field(None, max_length=4000)
    entry_plan: Optional[str] = Field(None, max_length=2000)
    exit_plan: Optional[str] = Field(None, max_length=2000)
    risk_notes: Optional[str] = Field(None, max_length=2000)
    risk_amount: Optional[float] = Field(None, ge=0)
    confidence: Optional[float] = Field(None, ge=0, le=1)
    emotion: Optional[str] = Field(None, max_length=120)
    journal_notes: Optional[str] = Field(None, max_length=4000)
    tags: Optional[str] = Field(None, max_length=400)

    # Simulation & attribution metadata
    is_simulated: Optional[bool] = False
    simulation_group: Optional[str] = Field(None, max_length=120)
    created_by: Optional[str] = Field(None, max_length=120)
    strategy_source: Optional[str] = Field(None, max_length=120)


class TradingLogIngestRequest(BaseModel):
    entries: List[TradingLogEntryPayload]


class TradingLogEntryUpdate(BaseModel):
    strategy_name: Optional[str] = Field(None, max_length=120)
    setup_name: Optional[str] = Field(None, max_length=120)
    thesis: Optional[str] = Field(None, max_length=4000)
    entry_plan: Optional[str] = Field(None, max_length=2000)
    exit_plan: Optional[str] = Field(None, max_length=2000)
    risk_notes: Optional[str] = Field(None, max_length=2000)
    risk_amount: Optional[float] = Field(None, ge=0)
    confidence: Optional[float] = Field(None, ge=0, le=1)
    emotion: Optional[str] = Field(None, max_length=120)
    journal_notes: Optional[str] = Field(None, max_length=4000)
    tags: Optional[str] = Field(None, max_length=400)
    is_simulated: Optional[bool] = None
    simulation_group: Optional[str] = Field(None, max_length=120)
    strategy_source: Optional[str] = Field(None, max_length=120)


class BrokerConnectorCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)
    provider: str = Field(..., min_length=2, max_length=40)
    credentials: Optional[str] = Field(None, max_length=4000)


class PortfolioStrategyRequest(BaseModel):
    scenario: str = Field("neutral", min_length=2, max_length=20)
    timeframe: str = Field("swing", min_length=2, max_length=20)
    notes: Optional[str] = Field(None, max_length=1000)


class JourneyRequest(BaseModel):
    trade_limit: int = Field(120, ge=20, le=500)
    include_simulated: bool = True


class SimulationGroupCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    hypothesis: Optional[str] = Field(None, max_length=4000)
    guardrail_max_positions: Optional[int] = Field(None, ge=0)
    guardrail_max_capital: Optional[float] = Field(None, ge=0)
    follow_up_date: Optional[datetime] = None
    auto_tasks: Optional[List[str]] = None


class SimulationGroupUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=120)
    hypothesis: Optional[str] = Field(None, max_length=4000)
    guardrail_max_positions: Optional[int] = Field(None, ge=0)
    guardrail_max_capital: Optional[float] = Field(None, ge=0)
    follow_up_date: Optional[datetime] = None
    auto_tasks: Optional[List[str]] = None
    status: Optional[str] = Field(None, pattern=r"^(active|archived)$")


class BulkPromoteRequest(BaseModel):
    entry_ids: Optional[List[int]] = None
    limit: Optional[int] = Field(None, gt=0, le=200)


def _generate_synthetic_symbol(db: Session) -> str:
    while True:
        candidate = f"PORT-{uuid.uuid4().hex[:6].upper()}"
        exists = db.query(models.Portfolio).filter(models.Portfolio.synthetic_symbol == candidate).first()
        if not exists:
            return candidate


def _extract_price_from_mapping(data: Optional[dict]) -> Optional[float]:
    if not isinstance(data, dict):
        return None
    for key in (
        "last_price",
        "regularMarketPrice",
        "currentPrice",
        "regularMarketLastClose",
        "previousClose",
        "open",
    ):
        value = data.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def _fetch_latest_price(ticker: str) -> float:
    ticker = ticker.upper()
    try:
        ticker_obj = yf.Ticker(ticker)
        price = None
        try:
            price = _extract_price_from_mapping(getattr(ticker_obj, "fast_info", None) or {})
        except Exception:  # pragma: no cover
            price = None
        if not price:
            info = getattr(ticker_obj, "info", {}) or {}
            price = _extract_price_from_mapping(info)
        if price and price > 0:
            return float(price)
    except Exception as exc:  # pragma: no cover
        logging.warning("Unable to fetch price for %s: %s", ticker, exc)
    return 100.0


def _seed_positions_from_targets(
    db: Session,
    portfolio_id: int,
    seeds: List[StartingPosition],
) -> int:
    seeded = 0
    for seed in seeds:
        ticker = seed.ticker.upper().strip()
        if not ticker:
            continue
        price = _fetch_latest_price(ticker)
        quantity = round(seed.target_value / price, 4)
        if quantity <= 0:
            continue
        position = _get_or_create_position(db, portfolio_id, ticker)
        position.quantity = quantity
        position.avg_cost = price
        position.current_price = price
        position.weight = 0.0
        position.notes = position.notes or "Seeded from ticker list"
        seeded += 1
    db.commit()
    return seeded


def _serialize_portfolio(portfolio: models.Portfolio) -> dict:
    return {
        "id": portfolio.id,
        "name": portfolio.name,
        "synthetic_symbol": portfolio.synthetic_symbol,
        "notes": portfolio.notes,
        "position_count": len(portfolio.positions),
        "updated_at": portfolio.updated_at.isoformat() if portfolio.updated_at else None,
    }


def _serialize_connector(connector: models.BrokerConnector) -> dict:
    return {
        "id": connector.id,
        "name": connector.name,
        "provider": connector.provider,
        "provider_label": BROKER_PROVIDERS.get(connector.provider, connector.provider.title()),
        "last_status": connector.last_status,
        "last_message": connector.last_message,
        "last_synced_at": connector.last_synced_at.isoformat() if connector.last_synced_at else None,
    }


def _serialize_trading_log_entry(entry: models.TradingLogEntry) -> dict:
    return {
        "id": entry.id,
        "portfolio_id": entry.portfolio_id,
        "executed_at": entry.executed_at.isoformat() if entry.executed_at else None,
        "action": entry.action,
        "ticker": entry.ticker,
        "quantity": entry.quantity,
        "price": entry.price,
        "source": entry.source,
        "strategy_name": entry.strategy_name,
        "setup_name": entry.setup_name,
        "thesis": entry.thesis,
        "entry_plan": entry.entry_plan,
        "exit_plan": entry.exit_plan,
        "risk_notes": entry.risk_notes,
        "risk_amount": entry.risk_amount,
        "confidence": entry.confidence,
        "emotion": entry.emotion,
        "journal_notes": entry.journal_notes,
        "tags": entry.tags,
        "is_simulated": entry.is_simulated,
        "simulation_group": entry.simulation_group,
        "created_by": entry.created_by,
        "strategy_source": entry.strategy_source,
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
        "updated_at": entry.updated_at.isoformat() if entry.updated_at else None,
    }


def _slugify(value: str) -> str:
    if not value:
        return ""
    value = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return value or f"group-{uuid.uuid4().hex[:4]}"


def _generate_unique_group_slug(db: Session, portfolio_id: int, seed: str) -> str:
    base = _slugify(seed) or f"group-{uuid.uuid4().hex[:4]}"
    slug = base
    suffix = 2
    while (
        db.query(models.SimulationGroup)
        .filter(
            models.SimulationGroup.portfolio_id == portfolio_id,
            models.SimulationGroup.slug == slug,
        )
        .first()
    ):
        slug = f"{base}-{suffix}"
        suffix += 1
    return slug


def _auto_tasks_to_string(tasks: Optional[List[str]]) -> str:
    if not tasks:
        return ""
    cleaned = [task.strip() for task in tasks if task and task.strip()]
    return "\n".join(cleaned)


def _auto_tasks_to_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _simulation_group_stats(db: Session, portfolio_id: int, slug: str) -> Dict[str, object]:
    if not slug:
        return {
            "simulated_entries": 0,
            "promoted_entries": 0,
            "simulated_notional": 0.0,
            "last_entry_at": None,
        }

    base_query = (
        db.query(models.TradingLogEntry)
        .filter(
            models.TradingLogEntry.portfolio_id == portfolio_id,
            models.TradingLogEntry.simulation_group == slug,
        )
    )

    simulated_query = base_query.filter(models.TradingLogEntry.is_simulated == True)  # noqa: E712
    promoted_query = base_query.filter(models.TradingLogEntry.is_simulated == False)  # noqa: E712

    simulated_entries = simulated_query.count()
    promoted_entries = promoted_query.count()
    simulated_notional = (
        simulated_query.with_entities(
            func.coalesce(
                func.sum(func.abs(models.TradingLogEntry.quantity * models.TradingLogEntry.price)),
                0.0,
            )
        ).scalar()
        or 0.0
    )

    latest = base_query.order_by(models.TradingLogEntry.updated_at.desc()).first()
    latest_ts = None
    if latest:
        timestamp = latest.updated_at or latest.executed_at or latest.created_at
        if timestamp:
            latest_ts = timestamp.isoformat()

    return {
        "simulated_entries": simulated_entries,
        "promoted_entries": promoted_entries,
        "simulated_notional": round(float(simulated_notional), 2),
        "last_entry_at": latest_ts,
    }


def _evaluate_guardrails(
    group: models.SimulationGroup,
    stats: Dict[str, object],
    incoming_entries: int = 0,
    incoming_notional: float = 0.0,
) -> Dict[str, object]:
    alerts: List[str] = []
    state = "ok"

    current_entries = stats.get("simulated_entries", 0)
    projected_entries = current_entries + incoming_entries
    entry_limit = group.guardrail_max_positions
    if entry_limit is not None:
        warn_threshold = max(int(entry_limit * 0.8), entry_limit - 1)
        if projected_entries > entry_limit:
            alerts.append(
                f"Sim trade cap {entry_limit} exceeded with projected {projected_entries}."
            )
            state = "blocked"
        elif projected_entries >= warn_threshold and state != "blocked":
            alerts.append(
                f"Sim trade count {projected_entries}/{entry_limit} approaching guardrail."
            )
            state = "warning"

    current_notional = stats.get("simulated_notional", 0.0) or 0.0
    projected_notional = current_notional + incoming_notional
    capital_limit = group.guardrail_max_capital
    if capital_limit is not None:
        warn_threshold = capital_limit * 0.8
        if projected_notional > capital_limit:
            alerts.append(
                f"Sim capital ${projected_notional:,.0f} exceeds cap ${capital_limit:,.0f}."
            )
            state = "blocked"
        elif projected_notional >= warn_threshold and state != "blocked":
            alerts.append(
                f"Sim capital ${projected_notional:,.0f}/${capital_limit:,.0f} near guardrail."
            )
            state = "warning"

    return {"state": state, "alerts": alerts}


def _serialize_simulation_group(
    group: models.SimulationGroup,
    db: Session,
    stats: Optional[Dict[str, object]] = None,
) -> dict:
    stats = stats or _simulation_group_stats(db, group.portfolio_id, group.slug)
    guardrail = _evaluate_guardrails(group, stats)
    return {
        "id": group.id,
        "portfolio_id": group.portfolio_id,
        "name": group.name,
        "slug": group.slug,
        "hypothesis": group.hypothesis,
        "status": group.status,
        "guardrail_max_positions": group.guardrail_max_positions,
        "guardrail_max_capital": group.guardrail_max_capital,
        "follow_up_date": group.follow_up_date.isoformat() if group.follow_up_date else None,
        "auto_tasks": _auto_tasks_to_list(group.auto_tasks),
        "created_at": group.created_at.isoformat() if group.created_at else None,
        "updated_at": group.updated_at.isoformat() if group.updated_at else None,
        "stats": stats,
        "guardrail": guardrail,
    }


def _get_simulation_group(
    db: Session,
    portfolio_id: int,
    group_id: int,
) -> models.SimulationGroup:
    group = (
        db.query(models.SimulationGroup)
        .filter(
            models.SimulationGroup.portfolio_id == portfolio_id,
            models.SimulationGroup.id == group_id,
        )
        .first()
    )
    if not group:
        raise HTTPException(status_code=404, detail="Simulation group not found")
    return group


def _ensure_guardrails_for_entries(
    db: Session,
    portfolio_id: int,
    entries: List[TradingLogEntryPayload],
) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, Dict[str, float]] = {}
    for entry in entries:
        if not entry.is_simulated:
            continue
        slug = (entry.simulation_group or "").strip()
        if not slug:
            continue
        bucket = grouped.setdefault(slug, {"count": 0, "notional": 0.0})
        bucket["count"] += 1
        bucket["notional"] += abs(entry.quantity * entry.price)

    alerts: Dict[str, Dict[str, object]] = {}
    if not grouped:
        return alerts

    for slug, info in grouped.items():
        group = (
            db.query(models.SimulationGroup)
            .filter(
                models.SimulationGroup.portfolio_id == portfolio_id,
                models.SimulationGroup.slug == slug,
            )
            .first()
        )
        if not group:
            raise HTTPException(status_code=400, detail=f"Simulation group '{slug}' not found")

        stats = _simulation_group_stats(db, portfolio_id, slug)
        guard = _evaluate_guardrails(
            group,
            stats,
            incoming_entries=info["count"],
            incoming_notional=info["notional"],
        )
        alerts[slug] = guard
        if guard["state"] == "blocked":
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Simulation guardrail triggered",
                    "slug": slug,
                    "alerts": guard["alerts"],
                },
            )

    return alerts

@router.get("/", response_class=JSONResponse)
def list_portfolios(db: Session = Depends(get_db)):
    portfolios = db.query(models.Portfolio).order_by(models.Portfolio.updated_at.desc()).all()
    return JSONResponse([_serialize_portfolio(p) for p in portfolios])


@router.post("/", response_class=JSONResponse)
def create_portfolio(payload: PortfolioCreate, db: Session = Depends(get_db)):
    existing = db.query(models.Portfolio).filter(models.Portfolio.name == payload.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Portfolio with this name already exists")

    synthetic_symbol = _generate_synthetic_symbol(db)
    new_portfolio = models.Portfolio(name=payload.name, notes=payload.notes, synthetic_symbol=synthetic_symbol)
    db.add(new_portfolio)
    db.commit()
    db.refresh(new_portfolio)
    seeded = 0
    if payload.starting_positions:
        seeded = _seed_positions_from_targets(db, new_portfolio.id, payload.starting_positions)
        db.refresh(new_portfolio)
    data = _serialize_portfolio(new_portfolio)
    data["seeded_positions"] = seeded
    return JSONResponse(data, status_code=201)


@router.get("/{portfolio_id}", response_class=JSONResponse)
def get_portfolio(portfolio_id: int, db: Session = Depends(get_db)):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    data = _serialize_portfolio(portfolio)
    data["positions"] = [
        {
            "id": pos.id,
            "ticker": pos.ticker,
            "quantity": pos.quantity,
            "avg_cost": pos.avg_cost,
            "current_price": pos.current_price,
            "sector": pos.sector,
            "weight": pos.weight,
            "notes": pos.notes,
        }
        for pos in portfolio.positions
    ]
    data["connectors"] = [_serialize_connector(conn) for conn in portfolio.broker_connectors]
    return JSONResponse(data)


def _get_or_create_position(db: Session, portfolio_id: int, ticker: str) -> models.PortfolioPosition:
    ticker = ticker.upper()
    position = (
        db.query(models.PortfolioPosition)
        .filter(
            models.PortfolioPosition.portfolio_id == portfolio_id,
            models.PortfolioPosition.ticker == ticker,
        )
        .first()
    )
    if not position:
        position = models.PortfolioPosition(
            portfolio_id=portfolio_id,
            ticker=ticker,
            quantity=0.0,
            avg_cost=0.0,
        )
        db.add(position)
        db.flush()
    return position


def _apply_trade(position: models.PortfolioPosition, action: str, quantity: float, price: float):
    action = action.lower()
    if action not in {"buy", "sell"}:
        raise ValueError("Action must be 'buy' or 'sell'")

    if action == "buy":
        new_qty = position.quantity + quantity
        total_cost = position.quantity * position.avg_cost + quantity * price
        position.quantity = new_qty
        position.avg_cost = total_cost / new_qty if new_qty else 0.0
    else:  # sell
        new_qty = max(position.quantity - quantity, 0.0)
        position.quantity = new_qty
        if new_qty == 0:
            position.avg_cost = 0.0

    position.current_price = price
    position.updated_at = datetime.utcnow()


def _update_equity_state(
    positions: Dict[str, Dict[str, float]],
    ticker: str,
    action: str,
    quantity: float,
    price: float,
) -> None:
    ticker = ticker.upper()
    record = positions.get(ticker)
    if not record:
        record = {"quantity": 0.0, "last_price": price}
        positions[ticker] = record

    if action.lower() == "buy":
        record["quantity"] += quantity
    else:
        record["quantity"] = max(record["quantity"] - quantity, 0.0)

    record["last_price"] = price


def _compute_equity(positions: Dict[str, Dict[str, float]]) -> float:
    return round(
        sum(state["quantity"] * state["last_price"] for state in positions.values()),
        2,
    )


def _build_equity_analytics(portfolio: models.Portfolio, db: Session) -> dict:
    entries = (
        db.query(models.TradingLogEntry)
        .filter(models.TradingLogEntry.portfolio_id == portfolio.id)
        .order_by(models.TradingLogEntry.executed_at.asc(), models.TradingLogEntry.id.asc())
        .all()
    )

    overall_positions: Dict[str, Dict[str, float]] = {}
    live_positions: Dict[str, Dict[str, float]] = {}
    sim_positions: Dict[str, Dict[str, float]] = {}
    overall_all = []
    overall_live = []
    overall_sim = []
    strategy_states: Dict[str, Dict[str, object]] = {}
    drawdown_points: List[Dict[str, float]] = []
    peak_equity = 0.0
    worst_drawdown_ratio = 0.0
    discipline_events: List[Dict[str, object]] = []
    discipline_ratio_sum = 0.0
    discipline_ratio_count = 0
    day_labels = list(calendar.day_abbr)
    hour_labels = [f"{start:02d}-{start + 4:02d}" for start in range(0, 24, 4)]
    heatmap_values = [[0.0 for _ in hour_labels] for _ in day_labels]

    for entry in entries:
        executed_at = entry.executed_at or entry.created_at or entry.updated_at or datetime.utcnow()
        timestamp = executed_at.isoformat()

        _update_equity_state(overall_positions, entry.ticker, entry.action, entry.quantity, entry.price)
        overall_all.append(
            {
                "timestamp": timestamp,
                "equity": _compute_equity(overall_positions),
                "is_simulated": entry.is_simulated,
                "ticker": entry.ticker,
                "action": entry.action,
            }
        )

        if entry.is_simulated:
            _update_equity_state(sim_positions, entry.ticker, entry.action, entry.quantity, entry.price)
            overall_sim.append(
                {
                    "timestamp": timestamp,
                    "equity": _compute_equity(sim_positions),
                    "ticker": entry.ticker,
                    "action": entry.action,
                }
            )
        else:
            _update_equity_state(live_positions, entry.ticker, entry.action, entry.quantity, entry.price)
            overall_live.append(
                {
                    "timestamp": timestamp,
                    "equity": _compute_equity(live_positions),
                    "ticker": entry.ticker,
                    "action": entry.action,
                }
            )

        strategy_key = (entry.strategy_name or entry.setup_name or "Unlabeled").strip() or "Unlabeled"
        if strategy_key not in strategy_states:
            strategy_states[strategy_key] = {
                "positions": {},
                "points": [],
                "entry_count": 0,
                "live_trades": 0,
                "simulated_trades": 0,
                "discipline_ratio_sum": 0.0,
                "discipline_ratio_count": 0,
            }

        strategy_state = strategy_states[strategy_key]
        _update_equity_state(strategy_state["positions"], entry.ticker, entry.action, entry.quantity, entry.price)
        strategy_equity = _compute_equity(strategy_state["positions"])
        strategy_state["points"].append(
            {
                "timestamp": timestamp,
                "equity": strategy_equity,
                "ticker": entry.ticker,
                "action": entry.action,
                "is_simulated": entry.is_simulated,
            }
        )
        strategy_state["entry_count"] += 1
        if entry.is_simulated:
            strategy_state["simulated_trades"] += 1
        else:
            strategy_state["live_trades"] += 1

        has_plan = bool((entry.entry_plan or "").strip()) and bool((entry.exit_plan or "").strip())
        has_risk = entry.risk_amount is not None
        logged_reflection = bool((entry.journal_notes or entry.emotion or "").strip())
        discipline_checks = [has_plan, has_risk, logged_reflection]
        checks_possible = len(discipline_checks)
        checks_completed = sum(1 for flag in discipline_checks if flag)
        if checks_possible == 0:
            checks_possible = 1
            checks_completed = 1
        entry_ratio = checks_completed / checks_possible
        discipline_ratio_sum += entry_ratio
        discipline_ratio_count += 1
        strategy_state["discipline_ratio_sum"] += entry_ratio
        strategy_state["discipline_ratio_count"] += 1

        missing_flags = []
        if not has_plan:
            missing_flags.append("Plan")
        if not has_risk:
            missing_flags.append("Risk")
        if not logged_reflection:
            missing_flags.append("Reflection")

        discipline_events.append(
            {
                "timestamp": timestamp,
                "score": round(entry_ratio, 4),
                "flags": missing_flags,
                "ticker": entry.ticker,
                "strategy": strategy_key,
                "is_simulated": entry.is_simulated,
            }
        )

        weekday_idx = executed_at.weekday()
        hour_idx = min(len(hour_labels) - 1, executed_at.hour // 4)
        gap_value = round(1 - entry_ratio, 4)
        heatmap_values[weekday_idx][hour_idx] += gap_value

        current_equity = overall_all[-1]["equity"] if overall_all else 0.0
        peak_equity = max(peak_equity, current_equity)
        drawdown_ratio = 0.0
        if peak_equity:
            drawdown_ratio = (current_equity - peak_equity) / peak_equity
        drawdown_points.append({"timestamp": timestamp, "value": round(drawdown_ratio, 6)})
        if drawdown_ratio < worst_drawdown_ratio:
            worst_drawdown_ratio = drawdown_ratio

    strategies_payload = [
        {
            "key": key,
            "label": key,
            "entry_count": state["entry_count"],
            "live_trades": state["live_trades"],
            "simulated_trades": state["simulated_trades"],
            "points": state["points"],
            "discipline_score": round(
                (state["discipline_ratio_sum"] / state["discipline_ratio_count"]) * 100,
                1,
            )
            if state["discipline_ratio_count"]
            else None,
        }
        for key, state in strategy_states.items()
    ]

    discipline_score = (
        round((discipline_ratio_sum / discipline_ratio_count) * 100, 1)
        if discipline_ratio_count
        else None
    )

    current_equity = overall_all[-1]["equity"] if overall_all else 0.0
    max_drawdown_pct = round(abs(worst_drawdown_ratio) * 100, 2)
    peak_equity = round(peak_equity, 2)

    heatmap_max = max((max(row) for row in heatmap_values), default=0)
    if not heatmap_max:
        heatmap_values = []

    overlays = {
        "drawdown": drawdown_points,
        "discipline_events": discipline_events[-200:],
    }

    return {
        "portfolio_id": portfolio.id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "entry_count": len(entries),
        "series": {
            "overall": overall_all,
            "live_only": overall_live,
            "simulated_only": overall_sim,
        },
        "strategies": strategies_payload,
        "notes": "Equity approximates mark-to-market using latest trade prices per ticker.",
        "insights": {
            "current_equity": current_equity,
            "peak_equity": peak_equity,
            "max_drawdown_pct": max_drawdown_pct,
            "discipline_score": discipline_score,
        },
        "overlays": overlays,
        "heatmap": {
            "day_labels": day_labels,
            "hour_labels": hour_labels,
            "values": heatmap_values,
        },
    }


@router.post("/{portfolio_id}/positions", response_class=JSONResponse)
def upsert_positions(
    portfolio_id: int,
    positions: List[PositionPayload],
    db: Session = Depends(get_db),
):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    updated = []
    for payload in positions:
        pos = _get_or_create_position(db, portfolio_id, payload.ticker)
        pos.quantity = payload.quantity
        pos.avg_cost = payload.avg_cost
        pos.current_price = payload.current_price
        pos.sector = payload.sector or ""
        pos.weight = payload.weight
        pos.notes = payload.notes or ""
        pos.updated_at = datetime.utcnow()
        updated.append(pos.ticker)

    db.commit()
    return JSONResponse({"success": True, "updated": updated})


@router.post("/{portfolio_id}/trading-log/ingest", response_class=JSONResponse)
def ingest_trading_logs(
    portfolio_id: int,
    request: TradingLogIngestRequest,
    db: Session = Depends(get_db),
):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    guardrail_alerts = _ensure_guardrails_for_entries(db, portfolio_id, request.entries)
    entries_written = 0
    for entry in request.entries:
        is_simulated = bool(entry.is_simulated)
        if not is_simulated:
            pos = _get_or_create_position(db, portfolio_id, entry.ticker)
            _apply_trade(pos, entry.action, entry.quantity, entry.price)

        log_entry = models.TradingLogEntry(
            portfolio_id=portfolio_id,
            action=entry.action.lower(),
            ticker=entry.ticker.upper(),
            quantity=entry.quantity,
            price=entry.price,
            executed_at=entry.executed_at or datetime.utcnow(),
            source=entry.source or "manual",
            raw_payload=entry.raw_payload or "",
            strategy_name=(entry.strategy_name or "").strip(),
            setup_name=(entry.setup_name or "").strip(),
            thesis=entry.thesis or "",
            entry_plan=entry.entry_plan or "",
            exit_plan=entry.exit_plan or "",
            risk_notes=entry.risk_notes or "",
            risk_amount=entry.risk_amount,
            confidence=entry.confidence,
            emotion=(entry.emotion or "").strip(),
            journal_notes=entry.journal_notes or "",
            tags=(entry.tags or "").strip(),
            is_simulated=is_simulated,
            simulation_group=(entry.simulation_group or "").strip(),
            created_by=(entry.created_by or "manual_import"),
            strategy_source=(entry.strategy_source or "trading_log"),
        )
        db.add(log_entry)
        entries_written += 1

    db.commit()
    return JSONResponse({
        "success": True,
        "entries": entries_written,
        "guardrail_alerts": guardrail_alerts,
    })


@router.get("/{portfolio_id}/trading-log", response_class=JSONResponse)
def list_trading_log_entries(
    portfolio_id: int,
    db: Session = Depends(get_db),
    simulation_group: Optional[str] = Query(None, max_length=120),
):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    query = db.query(models.TradingLogEntry).filter(models.TradingLogEntry.portfolio_id == portfolio_id)
    if simulation_group:
        query = query.filter(models.TradingLogEntry.simulation_group == simulation_group.strip())

    entries = (
        query.order_by(models.TradingLogEntry.executed_at.desc(), models.TradingLogEntry.id.desc())
        .limit(1000)
        .all()
    )
    return JSONResponse([_serialize_trading_log_entry(entry) for entry in entries])


@router.patch("/{portfolio_id}/trading-log/{entry_id}", response_class=JSONResponse)
def update_trading_log_entry(
    portfolio_id: int,
    entry_id: int,
    payload: TradingLogEntryUpdate,
    db: Session = Depends(get_db),
):
    entry = (
        db.query(models.TradingLogEntry)
        .filter(
            models.TradingLogEntry.portfolio_id == portfolio_id,
            models.TradingLogEntry.id == entry_id,
        )
        .first()
    )
    if not entry:
        raise HTTPException(status_code=404, detail="Log entry not found")

    update_data = payload.dict(exclude_unset=True)
    for key, value in update_data.items():
        if key in {"strategy_name", "setup_name", "emotion", "simulation_group", "strategy_source"} and value is not None:
            setattr(entry, key, value.strip())
        elif key == "tags" and value is not None:
            entry.tags = value.strip()
        elif key == "journal_notes" and value is not None:
            entry.journal_notes = value
        elif key == "thesis" and value is not None:
            entry.thesis = value
        elif key in {"entry_plan", "exit_plan", "risk_notes"} and value is not None:
            setattr(entry, key, value)
        elif key in {"risk_amount", "confidence"}:
            setattr(entry, key, value)
        elif key == "is_simulated" and value is not None:
            entry.is_simulated = bool(value)

    entry.updated_at = datetime.utcnow()
    db.commit()
    entry = (
        db.query(models.TradingLogEntry)
        .filter(
            models.TradingLogEntry.portfolio_id == portfolio_id,
            models.TradingLogEntry.id == entry_id,
        )
        .first()
    )
    if not entry:
        raise HTTPException(status_code=404, detail="Log entry not found")

    db.delete(entry)
    db.commit()
    return JSONResponse({"success": True})


@router.post("/{portfolio_id}/trading-log/{entry_id}/promote", response_class=JSONResponse)
def promote_simulated_entry(
    portfolio_id: int,
    entry_id: int,
    db: Session = Depends(get_db),
):
    entry = (
        db.query(models.TradingLogEntry)
        .filter(
            models.TradingLogEntry.portfolio_id == portfolio_id,
            models.TradingLogEntry.id == entry_id,
        )
        .first()
    )
    if not entry:
        raise HTTPException(status_code=404, detail="Log entry not found")

    if not entry.is_simulated:
        return JSONResponse(_serialize_trading_log_entry(entry))

    position = _get_or_create_position(db, portfolio_id, entry.ticker)
    _apply_trade(position, entry.action, entry.quantity, entry.price)

    entry.is_simulated = False
    entry.strategy_source = entry.strategy_source or "simulation_promoted"
    entry.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(entry)
    return JSONResponse(_serialize_trading_log_entry(entry))


@router.get("/{portfolio_id}/simulation-groups", response_class=JSONResponse)
def list_simulation_groups(portfolio_id: int, db: Session = Depends(get_db)):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    groups = (
        db.query(models.SimulationGroup)
        .filter(models.SimulationGroup.portfolio_id == portfolio_id)
        .order_by(models.SimulationGroup.created_at.asc())
        .all()
    )
    return JSONResponse([_serialize_simulation_group(group, db) for group in groups])


@router.post("/{portfolio_id}/simulation-groups", response_class=JSONResponse)
def create_simulation_group(
    portfolio_id: int,
    payload: SimulationGroupCreate,
    db: Session = Depends(get_db),
):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    slug = _generate_unique_group_slug(db, portfolio_id, payload.name)
    group = models.SimulationGroup(
        portfolio_id=portfolio_id,
        name=payload.name.strip(),
        slug=slug,
        hypothesis=(payload.hypothesis or "").strip(),
        guardrail_max_positions=payload.guardrail_max_positions,
        guardrail_max_capital=payload.guardrail_max_capital,
        follow_up_date=payload.follow_up_date,
        auto_tasks=_auto_tasks_to_string(payload.auto_tasks),
    )
    db.add(group)
    db.commit()
    db.refresh(group)
    return JSONResponse(_serialize_simulation_group(group, db))


@router.patch("/{portfolio_id}/simulation-groups/{group_id}", response_class=JSONResponse)
def update_simulation_group(
    portfolio_id: int,
    group_id: int,
    payload: SimulationGroupUpdate,
    db: Session = Depends(get_db),
):
    group = _get_simulation_group(db, portfolio_id, group_id)

    update_data = payload.dict(exclude_unset=True)
    if "name" in update_data and update_data["name"]:
        group.name = update_data["name"].strip()
    if "hypothesis" in update_data:
        group.hypothesis = (update_data["hypothesis"] or "").strip()
    if "guardrail_max_positions" in update_data:
        group.guardrail_max_positions = update_data["guardrail_max_positions"]
    if "guardrail_max_capital" in update_data:
        group.guardrail_max_capital = update_data["guardrail_max_capital"]
    if "follow_up_date" in update_data:
        group.follow_up_date = update_data["follow_up_date"]
    if "auto_tasks" in update_data:
        group.auto_tasks = _auto_tasks_to_string(update_data["auto_tasks"])
    if "status" in update_data and update_data["status"]:
        group.status = update_data["status"]

    group.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(group)
    return JSONResponse(_serialize_simulation_group(group, db))


@router.post(
    "/{portfolio_id}/simulation-groups/{group_id}/bulk-promote",
    response_class=JSONResponse,
)
def bulk_promote_simulation_group(
    portfolio_id: int,
    group_id: int,
    payload: BulkPromoteRequest = Body(default=BulkPromoteRequest()),
    db: Session = Depends(get_db),
):
    group = _get_simulation_group(db, portfolio_id, group_id)

    query = (
        db.query(models.TradingLogEntry)
        .filter(
            models.TradingLogEntry.portfolio_id == portfolio_id,
            models.TradingLogEntry.is_simulated == True,  # noqa: E712
            models.TradingLogEntry.simulation_group == group.slug,
        )
        .order_by(models.TradingLogEntry.executed_at.asc(), models.TradingLogEntry.id.asc())
    )

    if payload.entry_ids:
        query = query.filter(models.TradingLogEntry.id.in_(payload.entry_ids))
    elif payload.limit:
        query = query.limit(payload.limit)

    entries = query.all()
    if not entries:
        stats = _simulation_group_stats(db, portfolio_id, group.slug)
        guard = _evaluate_guardrails(group, stats)
        return JSONResponse({"promoted": 0, "guardrail": guard})

    promoted = 0
    for entry in entries:
        position = _get_or_create_position(db, portfolio_id, entry.ticker)
        _apply_trade(position, entry.action, entry.quantity, entry.price)
        entry.is_simulated = False
        entry.strategy_source = entry.strategy_source or "simulation_promoted"
        entry.updated_at = datetime.utcnow()
        promoted += 1

    db.commit()
    stats = _simulation_group_stats(db, portfolio_id, group.slug)
    guard = _evaluate_guardrails(group, stats)
    return JSONResponse({"promoted": promoted, "guardrail": guard})


@router.get("/{portfolio_id}/analytics/equity", response_class=JSONResponse)
def get_equity_curves(portfolio_id: int, db: Session = Depends(get_db)):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    analytics = _build_equity_analytics(portfolio, db)
    return JSONResponse(analytics)


@router.post("/{portfolio_id}/analytics/journey", response_class=JSONResponse)
def run_trading_journey(
    portfolio_id: int,
    payload: JourneyRequest = Body(default=JourneyRequest()),
    db: Session = Depends(get_db),
):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    analytics = _build_equity_analytics(portfolio, db)

    trade_query = (
        db.query(models.TradingLogEntry)
        .filter(models.TradingLogEntry.portfolio_id == portfolio_id)
        .order_by(models.TradingLogEntry.executed_at.desc(), models.TradingLogEntry.id.desc())
    )
    if not payload.include_simulated:
        trade_query = trade_query.filter(models.TradingLogEntry.is_simulated == False)  # noqa: E712
    entries = trade_query.limit(payload.trade_limit).all()

    trades_payload = [
        {
            "executed_at": (entry.executed_at or entry.created_at or entry.updated_at).isoformat() if (entry.executed_at or entry.created_at or entry.updated_at) else None,
            "action": entry.action,
            "ticker": entry.ticker,
            "quantity": entry.quantity,
            "price": entry.price,
            "strategy": entry.strategy_name or entry.setup_name,
            "is_simulated": entry.is_simulated,
            "thesis": entry.thesis,
            "notes": entry.journal_notes,
            "emotion": entry.emotion,
            "tags": entry.tags,
            "confidence": entry.confidence,
            "risk_amount": entry.risk_amount,
        }
        for entry in entries
    ][::-1]

    overall_series = analytics["series"].get("overall", [])
    overall_step = max(1, len(overall_series) // 200) if overall_series else 1
    equity_samples = overall_series[::overall_step]

    strategy_samples = []
    for strategy in analytics["strategies"]:
        points = strategy.get("points", [])
        step = max(1, len(points) // 50) if points else 1
        strategy_samples.append(
            {
                "label": strategy["label"],
                "entry_count": strategy["entry_count"],
                "live_trades": strategy["live_trades"],
                "simulated_trades": strategy["simulated_trades"],
                "points": points[::step] if points else [],
            }
        )

    context = {
        "trades": trades_payload,
        "strategies": strategy_samples,
        "equity_points": equity_samples,
    }

    analysis = gemini_analyzer.analyze_trading_journey(portfolio.name, context)
    if analysis.startswith("Error:"):
        raise HTTPException(status_code=500, detail=analysis)

    html = markdown.markdown(analysis, extensions=["tables", "fenced_code"])
    payload_record = {
        "html": html,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    return JSONResponse(payload_record)


@router.get("/{portfolio_id}/connectors", response_class=JSONResponse)
def list_connectors(portfolio_id: int, db: Session = Depends(get_db)):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    return JSONResponse([_serialize_connector(conn) for conn in portfolio.broker_connectors])


@router.post("/{portfolio_id}/connectors", response_class=JSONResponse)
def create_connector(
    portfolio_id: int,
    payload: BrokerConnectorCreate,
    db: Session = Depends(get_db),
):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    provider_key = payload.provider.lower()
    if provider_key not in BROKER_PROVIDERS:
        raise HTTPException(status_code=400, detail="Unknown broker provider")

    connector = models.BrokerConnector(
        portfolio_id=portfolio_id,
        name=payload.name.strip(),
        provider=provider_key,
        credentials=payload.credentials or "",
        last_status="configured",
        last_message="Connector saved. Run a sync once credentials are verified.",
    )
    db.add(connector)
    db.commit()
    db.refresh(connector)

    return JSONResponse(_serialize_connector(connector))


@router.delete("/{portfolio_id}/connectors/{connector_id}", response_class=JSONResponse)
def delete_connector(
    portfolio_id: int,
    connector_id: int,
    db: Session = Depends(get_db),
):
    connector = (
        db.query(models.BrokerConnector)
        .filter(
            models.BrokerConnector.portfolio_id == portfolio_id,
            models.BrokerConnector.id == connector_id,
        )
        .first()
    )
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")

    db.delete(connector)
    db.commit()
    return JSONResponse({"success": True})


@router.post("/{portfolio_id}/strategy", response_class=JSONResponse)
def run_portfolio_strategy(
    portfolio_id: int,
    payload: PortfolioStrategyRequest,
    db: Session = Depends(get_db),
):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    updated_signature = portfolio.updated_at.isoformat() if portfolio.updated_at else "never"
    notes_context = (payload.notes or portfolio.notes or "").strip()
    cache_fingerprint = f"{portfolio_id}:{updated_signature}:{payload.scenario}:{payload.timeframe}:{notes_context}"
    cache_key = "portfolio:" + hashlib.sha256(cache_fingerprint.encode()).hexdigest()

    cached = strategy_cache.get(cache_key)
    if cached:
        return JSONResponse({**cached, "cached": True})

    positions = [
        {
            "ticker": pos.ticker,
            "quantity": pos.quantity,
            "avg_cost": pos.avg_cost,
            "current_price": pos.current_price,
            "sector": pos.sector,
            "weight": pos.weight,
            "notes": pos.notes,
        }
        for pos in portfolio.positions
    ]

    strategy_markdown = gemini_analyzer.recommend_strategy_for_portfolio(
        portfolio_name=portfolio.name,
        positions=positions,
        scenario=payload.scenario,
        timeframe=payload.timeframe,
        synthetic_symbol=portfolio.synthetic_symbol,
        notes=payload.notes or portfolio.notes,
    )

    if strategy_markdown.startswith("Error:"):
        raise HTTPException(status_code=500, detail=strategy_markdown)

    html = markdown.markdown(strategy_markdown, extensions=["tables", "fenced_code"])
    payload_record = {
        "html": html,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    strategy_cache.set(cache_key, payload_record)
    return JSONResponse({**payload_record, "cached": False})


@router.post("/{portfolio_id}/ocr-upload", response_class=JSONResponse)
async def upload_portfolio_screenshot(
    portfolio_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    if file.content_type not in {"image/png", "image/jpeg", "image/jpg", "image/webp"}:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    processed_bytes, output_mime, preprocessing_meta, text_hint = ocr_pipeline.run_pipeline(
        contents,
        mime_type=file.content_type,
    )

    result = gemini_analyzer.extract_positions_from_image(
        processed_bytes,
        portfolio_hint=portfolio.name,
        mime_type=output_mime,
        text_hint=text_hint,
        preprocessing_notes=preprocessing_meta,
    )

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return JSONResponse(result)
