"""Contains various tx eq sweep functions
"""

import csv
import time
from datetime import datetime

from serdes_python_api.pcie6_lf.pcie6_lf import PCIE6_LF

REVISION = '$Revision: #2 $'
DATE = '$Date: 2024/05/07 $'


def tx_eq_sweep(evb: PCIE6_LF,
                lane: int,
                repetitions: int = 2,
                pause: float = 0.25,
                cm2_lo: int = 0,
                cm2_hi: int = 4,
                cm2_step: int = 1,
                cm1_lo: int = 0,
                cm1_step: int = 1,
                cm1_hi: int = 10,
                c0_lo: int = 0,
                c0_step: int = 1,
                c0_hi: int = 42,
                c1_lo: int = 0,
                c1_step: int = 1,
                c1_hi: int = 14) -> None:
    """Sweeps all cursors

    :param repetitions: Number of configurations at each cursor combination, defaults to 2
    :param pause: Pause time (seconds) between configurations, defaults to 0.25
    """

    with open(f"tx_eq_sweep_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv", 'w', newline='',
              encoding='UTF-8') as file:
        fields = ['lane', 'cm2', 'cm1', 'c0', 'c1', 'ber', 'date']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        iteration = 0
        print("\n******** Starting TX EQ Sweep ********")
        for a in range(cm2_lo, cm2_hi + 1, cm2_step):
            for b in range(cm1_lo, cm1_hi + 1, cm1_step):
                for c in range(c0_lo, c0_hi + 1, c0_step):
                    for d in range(c1_lo, c1_hi + 1, c1_step):
                        if a + b + c + d > 42:
                            continue
                        for _ in range(repetitions):
                            evb.sp_args.update({'tx_eq_L': 'custom', 'cm2_L': a, 'cm1_L': b,
                                                'c0_L': c, 'c1_L': d})
                            time.sleep(pause)
                            evb.initialize()
                            ber = evb.meas_ber(lane)
                            writer.writerow({'lane': lane, 'cm2': a, 'cm1': b, 'c0': c, 'c1': d,
                                             'ber': ber, 'date': datetime.now()})

                            iteration += 1
                            print(f"Iteration {iteration}, Lane: {lane}, cm2: {a}, cm1: {b}, "
                                  f"c0: {c}, c1: {d}, BER: {ber}")


def tx_eq_sweep_fixed(evb: PCIE6_LF,
                      lane: int,
                      repetitions: int = 2,
                      pause: float = 0.25,
                      cm2_lo: int = 0,
                      cm2_hi: int = 4,
                      cm2_step: int = 1,
                      cm1_lo: int = 0,
                      cm1_step: int = 1,
                      cm1_hi: int = 10,
                      c1_lo: int = 0,
                      c1_step: int = 1,
                      c1_hi: int = 14) -> None:
    """Sweeps all cursors, if sum is less than 42 the extra gets added to c0

    :param repetitions: Number of configurations at each cursor combination, defaults to 2
    :param pause: Pause time (seconds) between configurations, defaults to 0.25
    """

    with open(f"tx_eq_sweep_fixed_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv", 'w',
              newline='', encoding='UTF-8') as file:
        fields = ['lane', 'cm2', 'cm1', 'c0', 'c1', 'ber', 'date']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        iteration = 0
        iterations = int((cm2_hi - cm2_lo + 1) / cm2_step) * \
            int((cm1_hi - cm1_lo + 1) / cm1_step) * int((c1_hi - c1_lo + 1) / c1_step) * repetitions
        time_start = time.time()
        print("\n******** Starting TX EQ Sweep Fixed ********")
        for a in range(cm2_lo, cm2_hi + 1, cm2_step):
            for b in range(cm1_lo, cm1_hi + 1, cm1_step):
                for c in range(c1_lo, c1_hi + 1, c1_step):
                    for _ in range(repetitions):
                        evb.sp_args.update({'tx_eq_L': 'custom', 'cm2_L': a, 'cm1_L': b,
                                            'c0_L': 42 - a - b - c, 'c1_L': c})
                        time.sleep(pause)
                        evb.initialize()
                        ber = evb.meas_ber(lane)
                        writer.writerow({'lane': lane, 'cm2': a, 'cm1': b, 'c0': 42 - a - b - c,
                                         'c1': c, 'ber': ber, 'date': datetime.now()})

                        iteration += 1
                        time_elapsed = time.time() - time_start
                        time_average = time_elapsed / iteration
                        print(f"Iteration {iteration}/{iterations}, Lane: {lane}, cm2: {a}, cm1: "
                              f"{b}, c0: {c}, c1: {c}, BER: {ber}, Elapsed: "
                              f"{round(time_elapsed, 2)}s, Remaining: "
                              f"{round((time_average * (iterations - iteration)), 2)}s")


def tx_eq_sweep_preset(evb: PCIE6_LF, lane: int, repetitions: int = 5, pause: float = 0.25, presets:
                       list[str] | None = None) -> None:
    """Sweep cursors over presets

    :param repetitions: Number of configurations at each cursor combination, defaults to 2
    :param pause: Pause time (seconds) between configurations, defaults to 0.25
    :param presets: TX EQ preset, defaults to None
    """

    if presets is None:
        presets = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10']

    with open(f"tx_eq_sweep_preset_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv", 'w',
              newline='', encoding='UTF-8') as file:
        fields = ['lane', 'preset', 'ber', 'date']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        iterations = len(presets) * repetitions
        time_start = time.time()
        print("\n******** Starting TX EQ Sweep Preset ********")
        for i, preset in enumerate(presets):
            for r in range(repetitions):
                evb.sp_args.update({'tx_eq_L': preset})
                time.sleep(pause)
                evb.initialize()
                ber = evb.meas_ber(lane)
                writer.writerow({'lane': lane, 'preset': preset,
                                'ber': ber, 'date': datetime.now()})

                iteration = (r + 1) + (i * repetitions)
                time_elapsed = time.time() - time_start
                time_average = time_elapsed / iteration
                print(f"Iteration {iteration}/{iterations}, Lane: {lane}, Preset: {preset}, "
                      f"Elapsed: {round(time_elapsed, 2)}s, Remaining: "
                      f"{round((time_average * (iterations - iteration)), 2)}s")


def tx_eq_sweep_custom(evb: PCIE6_LF, lane: int, cursors: list[dict], repetitions: int = 2,
                       pause: float = 0.25):
    """Sweep cursors over custom values

    :param cursors: list of dictionaries with k = cursor (cm2, cm1, c0, or c1), v = value
    :param repetitions: Number of configurations at each cursor combination, defaults to 2
    :param pause: Pause time (seconds) between configurations, defaults to 0.25
    """

    with open(f"tx_eq_sweep_custom_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv", 'w',
              newline='', encoding='UTF-8') as file:
        fields = ['lane', 'cm2', 'cm1', 'c0', 'c1', 'ber', 'date']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        iterations = len(cursors) * repetitions
        time_start = time.time()
        print("\n******** Starting TX EQ Sweep Custom ********")
        for i, cursor_dict in enumerate(cursors):
            for r in range(repetitions):
                evb.sp_args.update(
                    {'tx_eq_L': 'custom', **{f'{k}_L': v for k, v in cursor_dict.items()}})
                time.sleep(pause)
                evb.initialize()
                ber = evb.meas_ber(lane)
                writer.writerow({'lane': lane, 'cm2': cursor_dict['cm2'],
                                 'cm1': cursor_dict['cm1'], 'c0': cursor_dict['c0'],
                                 'c1': cursor_dict['c1'], 'ber': ber, 'date': datetime.now()})

                iteration = (r + 1) + (i * repetitions)
                time_elapsed = time.time() - time_start
                time_average = time_elapsed / iteration
                print(f"Iteration {iteration}/{iterations}, Lane: {lane}, cm2: {cursor_dict['cm2']}"
                      f", cm1: {cursor_dict['cm1']}, c0: {cursor_dict['c0']}, c1: "
                      f"{cursor_dict['c1']}, BER: {ber}, Elapsed: {round(time_elapsed, 2)}s, "
                      f"Remaining: {round((time_average * (iterations - iteration)), 2)}s")
