import os
import sys
import importlib.util
from sqlalchemy.orm import Session
from webapp.database import SessionLocal, engine
from webapp import models

def migrate_ticker_lists():
    # Path to original ticker_lists.py
    ticker_lists_path = "stock_charts_10k10q/ticker_lists.py"

    if not os.path.exists(ticker_lists_path):
        print(f"Error: {ticker_lists_path} not found.")
        return

    # Load module dynamically
    spec = importlib.util.spec_from_file_location("ticker_lists", ticker_lists_path)
    ticker_lists_module = importlib.util.module_from_spec(spec)
    sys.modules["ticker_lists_mig"] = ticker_lists_module
    spec.loader.exec_module(ticker_lists_module)

    db = SessionLocal()

    # Clear existing data (optional, for clean slate)
    # db.query(models.Ticker).delete()
    # db.query(models.TickerList).delete()
    # db.commit()

    count = 0

    # Iterate through attributes
    for name in dir(ticker_lists_module):
        if name.startswith("__") or callable(getattr(ticker_lists_module, name)):
            continue

        val = getattr(ticker_lists_module, name)
        if isinstance(val, list):
            # Create list
            print(f"Migrating list: {name} with {len(val)} tickers")

            # Check if list exists
            existing_list = db.query(models.TickerList).filter(models.TickerList.name == name).first()
            if not existing_list:
                is_watch = (name == "watch_list")
                new_list = models.TickerList(name=name, is_watch_list=is_watch)
                db.add(new_list)
                db.commit()
                db.refresh(new_list)
                list_id = new_list.id
            else:
                list_id = existing_list.id

            # Add tickers
            for ticker_symbol in val:
                if not isinstance(ticker_symbol, str): continue

                # Check if ticker exists in list
                existing_ticker = db.query(models.Ticker).filter(
                    models.Ticker.list_id == list_id,
                    models.Ticker.symbol == ticker_symbol
                ).first()

                if not existing_ticker:
                    new_ticker = models.Ticker(list_id=list_id, symbol=ticker_symbol)
                    db.add(new_ticker)
                    count += 1

            db.commit()

    print(f"Migration complete. Added {count} tickers.")
    db.close()

if __name__ == "__main__":
    migrate_ticker_lists()
