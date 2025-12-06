from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
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
