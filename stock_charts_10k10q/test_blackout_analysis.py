"""Unit tests for blackout_analysis (synthetic data only — no network, no GUI)."""

import unittest
from datetime import date

import pandas as pd

from blackout_analysis import (
    get_blackout_periods,
    compute_quarter_gains,
    compute_compounded_curves,
    compute_daily_equity_curves,
    split_blackout_values,
    extend_result_horizon,
)
import math


def make_history(start, end, overrides=None, default_close=100.0):
    """Build a business-day price frame with optional per-date close overrides."""
    idx = pd.bdate_range(start, end)
    closes = {d.date(): default_close for d in idx}
    if overrides:
        for d, c in overrides.items():
            closes[d] = float(c)
    dates = sorted(closes.keys())
    stamps = pd.to_datetime([d.isoformat() for d in dates], utc=True)
    return pd.DataFrame({'Date': stamps, 'Close': [closes[d] for d in dates]})


def period(end, start_days=30):
    return get_blackout_periods([{'date': end}], start_days)[0]


class TestGetBlackoutPeriods(unittest.TestCase):
    def test_label_and_dates(self):
        # Feb 7 announcement -> calendar Q1 label, start 30 days earlier
        p = period(date(2024, 2, 7))
        self.assertEqual(p['label'], '2024 Q1')
        self.assertEqual(p['blackout_end'], date(2024, 2, 7))
        self.assertEqual(p['blackout_start'], date(2024, 1, 8))

    def test_chronological_order(self):
        items = [{'date': date(2024, 5, 8)}, {'date': date(2024, 2, 7)}]
        periods = get_blackout_periods(items)
        self.assertEqual([p['blackout_end'] for p in periods],
                         [date(2024, 2, 7), date(2024, 5, 8)])


class TestComputeQuarterGains(unittest.TestCase):
    def test_horizon_rolls_weekend_forward(self):
        # Earnings Friday 2024-04-05. N=1 targets Saturday -> Monday 2024-04-08.
        hist = make_history('2024-04-01', '2024-04-30',
                            overrides={date(2024, 4, 5): 100.0,
                                       date(2024, 4, 8): 110.0})
        rows = compute_quarter_gains(hist, [period(date(2024, 4, 5))],
                                     horizons=[1, 3], as_of=date(2024, 4, 30))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['baseline_date'], date(2024, 4, 5))
        self.assertAlmostEqual(row['gains'][1], 0.10)   # Sat target -> Mon close
        # N=3 targets Monday 2024-04-08 (a trading day) -> same close
        self.assertAlmostEqual(row['gains'][3], 0.10)

    def test_earnings_on_weekend_baseline_rolls_forward(self):
        # Earnings Saturday 2024-04-06 -> baseline Monday 2024-04-08.
        hist = make_history('2024-04-01', '2024-04-30',
                            overrides={date(2024, 4, 8): 100.0,
                                       date(2024, 4, 9): 105.0})
        rows = compute_quarter_gains(hist, [period(date(2024, 4, 6))],
                                     horizons=[1], as_of=date(2024, 4, 30))
        self.assertEqual(rows[0]['baseline_date'], date(2024, 4, 8))
        # N=1 targets Sunday -> Monday, the baseline session itself -> 0%
        self.assertAlmostEqual(rows[0]['gains'][1], 0.0)

    def test_partial_windows_excluded_per_horizon(self):
        # Earnings 2024-04-08; as_of 2024-05-01 completes 1d..21d but not 42d.
        hist = make_history('2024-01-01', '2024-04-30',
                            overrides={date(2024, 4, 8): 100.0})
        rows = compute_quarter_gains(hist, [period(date(2024, 4, 8))],
                                     horizons=[1, 21, 42], as_of=date(2024, 5, 1))
        self.assertIsNotNone(rows[0]['gains'][1])
        self.assertIsNotNone(rows[0]['gains'][21])   # target 2024-04-29 <= as_of
        self.assertIsNone(rows[0]['gains'][42])      # target 2024-05-20 > as_of

    def test_exit_rolled_past_as_of_is_incomplete(self):
        # Target lands on a Saturday AFTER as_of (same Saturday): the next
        # trading day hasn't happened, so the window is incomplete.
        hist = make_history('2024-04-01', '2024-04-30',
                            overrides={date(2024, 4, 8): 100.0})
        # N=5 targets Saturday 2024-04-13; as_of is that Saturday.
        rows = compute_quarter_gains(hist, [period(date(2024, 4, 8))],
                                     horizons=[5], as_of=date(2024, 4, 13))
        self.assertIsNone(rows[0]['gains'][5])

    def test_quarter_without_price_data_is_skipped(self):
        # Earnings far beyond the price data -> no baseline session -> dropped.
        hist = make_history('2024-01-01', '2024-06-30')
        rows = compute_quarter_gains(hist, [period(date(2024, 12, 11))],
                                     horizons=[1], as_of=date(2025, 6, 30))
        self.assertEqual(rows, [])

    def test_baseline_in_future_is_skipped(self):
        hist = make_history('2024-01-01', '2024-06-30')
        rows = compute_quarter_gains(hist, [period(date(2024, 4, 8))],
                                     horizons=[1], as_of=date(2024, 3, 31))
        self.assertEqual(rows, [])

    def test_entry_start_buys_at_blackout_start(self):
        # Earnings 2024-05-08, blackout starts 30 days earlier: 2024-04-08.
        # entry='start' anchors baseline AND the N-day exit on the start.
        hist = make_history('2024-01-01', '2024-06-30',
                            overrides={date(2024, 4, 8): 100.0,   # start close (buy)
                                       date(2024, 4, 15): 110.0,  # start+7d exit
                                       date(2024, 5, 8): 200.0,   # earnings close
                                       date(2024, 5, 15): 180.0}) # end+7d (unused here)
        rows = compute_quarter_gains(hist, [period(date(2024, 5, 8))],
                                     horizons=[7], as_of=date(2024, 6, 30),
                                     entry='start')
        self.assertEqual(rows[0]['baseline_date'], date(2024, 4, 8))
        self.assertAlmostEqual(rows[0]['baseline_close'], 100.0)
        self.assertAlmostEqual(rows[0]['gains'][7], 0.10)  # 110/100 - 1

        # Default entry='end' buys at the earnings-day close instead
        rows_end = compute_quarter_gains(hist, [period(date(2024, 5, 8))],
                                         horizons=[7], as_of=date(2024, 6, 30))
        self.assertEqual(rows_end[0]['baseline_date'], date(2024, 5, 8))
        self.assertAlmostEqual(rows_end[0]['gains'][7], -0.10)  # 180/200 - 1


class TestComputeCompoundedCurves(unittest.TestCase):
    @staticmethod
    def two_quarter_gains():
        hist = make_history('2024-01-01', '2024-06-30',
                            overrides={date(2024, 2, 7): 100.0,   # Q1 baseline
                                       date(2024, 2, 14): 110.0,  # Q1 +7d  -> +10%
                                       date(2024, 5, 8): 200.0,   # Q2 baseline
                                       date(2024, 5, 15): 180.0}) # Q2 +7d  -> -10%
        periods = [period(date(2024, 2, 7)), period(date(2024, 5, 8))]
        return compute_quarter_gains(hist, periods, horizons=[7],
                                     as_of=date(2024, 6, 30))

    def test_compounding_math(self):
        # +10% then -10% compounds to -1%, not 0%
        curves = compute_compounded_curves(self.two_quarter_gains(), horizons=[7])
        self.assertEqual(curves[7]['labels'], ['2024 Q1', '2024 Q2'])
        self.assertAlmostEqual(curves[7]['values'][0], 10.0)
        self.assertAlmostEqual(curves[7]['values'][1], -1.0)
        self.assertAlmostEqual(curves[7]['final'], -1.0)
        self.assertEqual(curves[7]['count'], 2)

    def test_incomplete_windows_excluded_from_curve(self):
        rows = self.two_quarter_gains()
        rows[1]['gains'][7] = None  # latest quarter's 7d window not elapsed
        curves = compute_compounded_curves(rows, horizons=[7])
        self.assertEqual(curves[7]['labels'], ['2024 Q1'])
        self.assertAlmostEqual(curves[7]['final'], 10.0)


class TestComputeDailyEquityCurves(unittest.TestCase):
    @staticmethod
    def two_quarter_setup():
        hist = make_history('2024-01-01', '2024-06-30',
                            overrides={date(2024, 2, 7): 100.0,   # Q1 baseline (buy)
                                       date(2024, 2, 9): 105.0,   # mid-window mark
                                       date(2024, 2, 14): 110.0,  # Q1 +7d exit -> +10%
                                       date(2024, 5, 8): 200.0,   # Q2 baseline (buy)
                                       date(2024, 5, 15): 180.0}) # Q2 +7d exit -> -10%
        periods = [period(date(2024, 2, 7)), period(date(2024, 5, 8))]
        rows = compute_quarter_gains(hist, periods, horizons=[7],
                                     as_of=date(2024, 6, 30))
        return hist, rows

    @staticmethod
    def value_on(curve, d):
        return curve['values'][curve['dates'].index(d)]

    def test_daily_marking_and_cash_flat(self):
        hist, rows = self.two_quarter_setup()
        curves = compute_daily_equity_curves(hist, rows, horizons=[7],
                                             as_of=date(2024, 6, 30))
        c = curves[7]
        # Starts at 0% on the first baseline date
        self.assertEqual(c['dates'][0], date(2024, 2, 7))
        self.assertAlmostEqual(c['values'][0], 0.0)
        # Marked daily while holding: 105/100 -> +5%
        self.assertAlmostEqual(self.value_on(c, date(2024, 2, 9)), 5.0)
        # Exit day: +10%
        self.assertAlmostEqual(self.value_on(c, date(2024, 2, 14)), 10.0)
        # Cash between windows: flat at +10% in mid-March
        self.assertAlmostEqual(self.value_on(c, date(2024, 3, 15)), 10.0)
        # Final value equals compounded +10% then -10% -> -1%
        self.assertAlmostEqual(c['values'][-1], -1.0)

    def test_incomplete_window_held_live(self):
        # as_of lands inside the second window: position still open, marked
        # daily at the available closes (200 -> 190 mid-window = -1% overall).
        hist = make_history('2024-01-01', '2024-05-10',
                            overrides={date(2024, 2, 7): 100.0,
                                       date(2024, 2, 14): 110.0,
                                       date(2024, 5, 8): 200.0,
                                       date(2024, 5, 9): 190.0})
        periods = [period(date(2024, 2, 7)), period(date(2024, 5, 8))]
        rows = compute_quarter_gains(hist, periods, horizons=[7],
                                     as_of=date(2024, 5, 10))
        self.assertIsNone(rows[1]['gains'][7])  # window not complete yet
        curves = compute_daily_equity_curves(hist, rows, horizons=[7],
                                             as_of=date(2024, 5, 10))
        c = curves[7]
        self.assertEqual(c['dates'][-1], date(2024, 5, 9))  # last session <= as_of
        # 1.1 * (190/200) - 1 = 4.5%
        self.assertAlmostEqual(c['values'][-1], 4.5)

    def test_overlapping_windows_skip_rebuy(self):
        # Earnings 13 days apart with N=14: the second buy arrives one day
        # before the first exit -> the buy signal is SKIPPED and the
        # position runs to its own exit (long-hold semantics).
        hist = make_history('2024-01-01', '2024-04-30',
                            overrides={date(2024, 2, 7): 100.0,   # buy #1
                                       date(2024, 2, 13): 120.0,  # +20% mid-window
                                       date(2024, 2, 20): 110.0,  # skipped buy signal
                                       date(2024, 2, 21): 115.0,  # own 14d exit
                                       date(2024, 3, 5): 121.0})  # other window's exit (never used)
        periods = [period(date(2024, 2, 7)), period(date(2024, 2, 20))]
        rows = compute_quarter_gains(hist, periods, horizons=[14],
                                     as_of=date(2024, 4, 30))
        curves = compute_daily_equity_curves(hist, rows, horizons=[14],
                                             as_of=date(2024, 4, 30))
        c = curves[14]
        self.assertAlmostEqual(self.value_on(c, date(2024, 2, 13)), 20.0)
        # Held through the skipped signal (110/100)
        self.assertAlmostEqual(self.value_on(c, date(2024, 2, 20)), 10.0)
        # Exits at its OWN 14-day exit (2/21) -> +15%, then cash to the end
        self.assertAlmostEqual(self.value_on(c, date(2024, 2, 21)), 15.0)
        self.assertAlmostEqual(c['values'][-1], 15.0)

    def test_long_hold_spans_multiple_quarters(self):
        # N=120 days with quarterly earnings: the second buy signal (5/8)
        # arrives while still holding -> skipped; exit at 2/7+120d = 6/6.
        hist = make_history('2024-01-01', '2024-08-31',
                            overrides={date(2024, 2, 7): 100.0,   # buy
                                       date(2024, 5, 8): 130.0,   # skipped signal
                                       date(2024, 6, 6): 120.0})  # 120d exit
        periods = [period(date(2024, 2, 7)), period(date(2024, 5, 8))]
        rows = compute_quarter_gains(hist, periods, horizons=[120],
                                     as_of=date(2024, 8, 31))
        curves = compute_daily_equity_curves(hist, rows, horizons=[120],
                                             as_of=date(2024, 8, 31))
        c = curves[120]
        self.assertAlmostEqual(self.value_on(c, date(2024, 5, 8)), 30.0)  # marked while holding
        self.assertAlmostEqual(self.value_on(c, date(2024, 6, 6)), 20.0)  # own exit
        self.assertAlmostEqual(c['values'][-1], 20.0)                     # cash after


class TestFigureBuilders(unittest.TestCase):
    def test_subset_and_empty_horizons_render(self):
        # Checkbox filtering passes a horizon subset (possibly empty) to the
        # figure builders; both must render without error either way.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from blackout_analysis import (build_blackout_figure,
                                       build_daily_equity_figure,
                                       compute_compounded_curves)
        hist, rows = TestComputeDailyEquityCurves.two_quarter_setup()
        curves = compute_compounded_curves(rows, horizons=[7])
        daily = compute_daily_equity_curves(hist, rows, horizons=[7],
                                            as_of=date(2024, 6, 30))
        # Raw price series for the buy-and-hold overlay
        idx = pd.bdate_range('2024-01-01', '2024-06-30')
        overlay = {'dates': [d.date() for d in idx],
                   'closes': [100.0 + i for i in range(len(idx))]}
        for hor in ([7], []):
            for scatter in (False, True):
                fig_q = build_blackout_figure('TEST', rows, curves,
                                              horizons=hor, scatter=scatter,
                                              overlay_series=overlay)
                fig_d = build_daily_equity_figure('TEST', daily,
                                                  horizons=hor, scatter=scatter,
                                                  overlay_series=overlay)
                self.assertIsNotNone(fig_q)
                self.assertIsNotNone(fig_d)
                plt.close(fig_q)
                plt.close(fig_d)

    def test_blackout_vlines_drawn(self):
        # With show_blackout_lines: 2 quarters -> 2 end lines (both in range)
        # + 1 start line (the first quarter's start precedes the curve's
        # first day, so it is skipped to avoid stretching the x-axis).
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from blackout_analysis import (build_daily_equity_figure,
                                       compute_compounded_curves)
        hist, rows = TestComputeDailyEquityCurves.two_quarter_setup()
        daily = compute_daily_equity_curves(hist, rows, horizons=[7],
                                            as_of=date(2024, 6, 30))
        intervals = [(r['blackout_start'], r['blackout_end']) for r in rows]

        fig0 = build_daily_equity_figure('TEST', daily, horizons=[7])
        base_count = len(fig0.axes[0].get_lines())
        plt.close(fig0)

        fig1 = build_daily_equity_figure('TEST', daily, horizons=[7],
                                         show_blackout_lines=True,
                                         blackout_intervals=intervals)
        ax = fig1.axes[0]
        added = len(ax.get_lines()) - base_count
        self.assertEqual(added, 3)  # 2 end lines + 1 in-range start line
        green_ends = [l for l in ax.get_lines()
                       if l.get_color() == 'green' and l.get_linestyle() == ':']
        self.assertEqual(len(green_ends), 2)
        self.assertIn('Blackout start', [t.get_text() for t in ax.get_legend().get_texts()])
        self.assertIn('Blackout end', [t.get_text() for t in ax.get_legend().get_texts()])
        plt.close(fig1)


class TestSplitBlackoutValues(unittest.TestCase):
    def test_blackout_days_highlighted_end_day_joins(self):
        dates = [date(2024, 1, 8 + i) for i in range(5)]  # Jan 8..12
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        # Blackout Jan 9 -> Jan 11: Jan 9 and 10 are inside [start, end);
        # the end (earnings/buy) day belongs to BOTH series as the join.
        normal, yellow = split_blackout_values(
            dates, values, [(date(2024, 1, 9), date(2024, 1, 11))])
        self.assertEqual(normal[0], 1.0)
        self.assertTrue(math.isnan(normal[1]))
        self.assertTrue(math.isnan(normal[2]))
        self.assertEqual(normal[3], 4.0)   # end day kept in normal
        self.assertEqual(normal[4], 5.0)
        self.assertTrue(math.isnan(yellow[0]))
        self.assertEqual(yellow[1], 2.0)
        self.assertEqual(yellow[2], 3.0)
        self.assertEqual(yellow[3], 4.0)   # end day joins the highlight
        self.assertTrue(math.isnan(yellow[4]))

    def test_multiple_intervals(self):
        dates = [date(2024, 1, 1), date(2024, 2, 1), date(2024, 2, 15),
                 date(2024, 3, 1)]
        values = [10.0, 20.0, 30.0, 40.0]
        normal, yellow = split_blackout_values(
            dates, values, [(date(2024, 1, 15), date(2024, 2, 5)),
                            (date(2024, 2, 20), date(2024, 2, 25))])
        self.assertEqual(normal[0], 10.0)          # before first blackout
        self.assertTrue(math.isnan(normal[1]))     # Feb 1 inside first
        self.assertEqual(normal[2], 30.0)          # between blackouts
        self.assertEqual(normal[3], 40.0)
        self.assertTrue(math.isnan(yellow[0]))
        self.assertEqual(yellow[1], 20.0)
        self.assertTrue(math.isnan(yellow[2]))
        self.assertTrue(math.isnan(yellow[3]))

    def test_no_intervals_all_normal(self):
        normal, yellow = split_blackout_values(
            [date(2024, 1, 1), date(2024, 1, 2)], [7.0, 8.0], [])
        self.assertEqual(normal, [7.0, 8.0])
        self.assertTrue(all(math.isnan(v) for v in yellow))


class TestExtendResultHorizon(unittest.TestCase):
    @staticmethod
    def build_result():
        """Minimal analyze_ticker()-shaped result over synthetic data."""
        hist, rows = TestComputeDailyEquityCurves.two_quarter_setup()
        periods = [period(date(2024, 2, 7)), period(date(2024, 5, 8))]
        modes = {}
        for entry in ('end', 'start'):
            qg = compute_quarter_gains(hist, periods, horizons=[7],
                                       as_of=date(2024, 6, 30), entry=entry)
            modes[entry] = {
                'quarter_gains': qg,
                'curves': compute_compounded_curves(qg, horizons=[7]),
                'daily_curves': compute_daily_equity_curves(
                    hist, qg, horizons=[7], as_of=date(2024, 6, 30)),
                'blackout_intervals': [(r['blackout_start'], r['blackout_end'])
                                       for r in qg],
            }
        # price_series must mirror the history the modes were computed from
        ps = {'dates': pd.to_datetime(hist['Date'], utc=True).dt.date.tolist(),
              'closes': hist['Close'].tolist()}
        return {'ticker': 'TEST', 'periods': periods, 'modes': modes,
                'price_series': ps, 'as_of': date(2024, 6, 30),
                'quarters_analyzed': len(modes['end']['quarter_gains'])}

    def test_extends_with_custom_horizon(self):
        result = self.build_result()
        extend_result_horizon(result, 21)
        for entry in ('end', 'start'):
            mode = result['modes'][entry]
            self.assertIn(21, mode['curves'])
            self.assertIn(21, mode['daily_curves'])
            for row in mode['quarter_gains']:
                self.assertIn(21, row['gains'])
        # Q2 baseline 5/8 @200, +21d lands on a default-100 day -> -50%
        row2 = result['modes']['end']['quarter_gains'][1]
        self.assertAlmostEqual(row2['gains'][21], -0.5)
        # End-mode curve: Q1 +0% (21d lands on a default-100 day) then -50%
        self.assertAlmostEqual(result['modes']['end']['curves'][21]['final'],
                               -50.0)
        self.assertTrue(result['modes']['end']['daily_curves'][21]['values'])

    def test_existing_horizon_is_unchanged(self):
        result = self.build_result()
        before = result['modes']['end']['curves'][7]['final']
        extend_result_horizon(result, 7)  # already present -> no-op
        self.assertEqual(result['modes']['end']['curves'][7]['final'], before)

    def test_invalid_n_rejected(self):
        result = self.build_result()
        extend_result_horizon(result, 0)
        extend_result_horizon(result, -5)
        self.assertNotIn(0, result['modes']['end']['curves'])


if __name__ == '__main__':
    unittest.main()
