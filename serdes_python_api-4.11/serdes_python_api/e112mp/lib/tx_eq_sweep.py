"""Contains tx eq sweep functions
"""

import csv
import time
from datetime import datetime

from serdes_python_api.shared.shared import Shared

REVISION = '$Revision: #4 $'
DATE = '$Date: 2024/06/25 $'

def tx_eq_sweep(evb: Shared, lane: int,
                repetitions: int = 2,
                pause: float = 0.25,
                cm3_lo: int = 0,
                cm3_hi: int = 4,
                cm3_step: int = 1,
                cm2_lo: int = 0,
                cm2_hi: int = 8,
                cm2_step: int = 1,
                cm1_lo: int = 0,
                cm1_step: int = 1,
                cm1_hi: int = 22,
                c0_lo: int = 0,
                c0_step: int = 1,
                c0_hi: int = 63,
                c1_lo: int = 0,
                c1_step: int = 1,
                c1_hi: int = 13):
     """Contains TX Eq Sweep function
     """
     with open(f"tx_eq_sweep_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv", 'w', newline='',
              encoding='UTF-8') as file:
        fields = ['lane', 'cm3','cm2', 'cm1', 'c0', 'c1', 'errors', 'time', 'nbits', 'ber', 'date']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        print("\n******** Starting TX EQ Sweep ********")
        for a in range(cm3_lo, cm3_hi + 1, cm3_step):
            for b in range(cm2_lo, cm2_hi + 1, cm2_step):
                for c in range(cm1_lo, cm1_hi + 1, cm1_step):
                    for d in range(c0_lo, c0_hi + 1, c0_step):
                        for e in range(c1_lo, c1_hi + 1, c1_step):
                            if a + b + c + d + e > 63:
                                continue
                            for _ in range(repetitions):
                                evb.sp_args.update({'tx_eq_L': 'custom', 'cm3_L': a, 'cm2_L': b, 'cm1_L': c,
                                                    'c0_L': d, 'c1_L': e})
                                time.sleep(pause)
                                evb.initialize()
                                ber = evb.meas_ber(lane=[lane])
                                writer.writerow({'lane': lane, 'cm3': a, 'cm2': b,
                                                'cm1': c, 'c0': d, 'c1': e, 'ber': ber,
                                                 'date': datetime.now()})
                                iteration += 1
                                print(f"Iteration {iteration}, Lane: {lane}, cm3: {a}, cm2: {b}, "
                                  f"cm1: {c}, c0: {d}, c1: {e}, BER: {ber}")


def tx_eq_sweep_fixed(evb: Shared, lane: int,
                      repetitions: int = 2,
                      pause: float = 0.25,
                      cm3_lo: int = 0,
                      cm3_hi: int = 4,
                      cm3_step: int = 1,
                      cm2_lo: int = 0,
                      cm2_hi: int = 8,
                      cm2_step: int = 1,
                      cm1_lo: int = 0,
                      cm1_step: int = 1,
                      cm1_hi: int = 22,
                      c1_lo: int = 0,
                      c1_step: int = 1,
                      c1_hi: int = 13):
     """Contains TX Eq fixed Sweep function
     """
     with open(f"tx_eq_sweep_fixed_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv", 'w',
              newline='', encoding='UTF-8') as file:
        fields = ['lane', 'cm3', 'cm2', 'cm1', 'c0', 'c1', 'errors', 'time', 'nbits', 'ber', 'date']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        iteration = 0
        iterations = int((cm3_hi - cm3_lo + 1) / cm3_step) * int((cm2_hi - cm2_lo + 1) / cm2_step) * \
            int((cm1_hi - cm1_lo + 1) / cm1_step) * int((c1_hi - c1_lo + 1) / c1_step) * repetitions
        time_start = time.time()
        print("\n******** Starting TX EQ Sweep ********")
        for a in range(cm3_lo, cm3_hi + 1, cm3_step):
            for b in range(cm2_lo, cm2_hi + 1, cm2_step):
                for c in range(cm1_lo, cm1_hi + 1, cm1_step):
                    for d in range(c1_lo, c1_hi + 1, c1_step):
                        for _ in range(repetitions):
                            evb.sp_args.update({'tx_eq_L': 'custom', 'cm3_L': a, 'cm2_L': b, 'cm1_L': c,
                                                'c0_L': 63 - a - b - c - d, 'c1_L': d})
                            time.sleep(pause)
                            evb.initialize()
                            ber = evb.meas_ber(lane=[lane])
                            writer.writerow({'lane': lane, 'cm3': a, 'cm2': b,
                                             'cm1': c, 'c0':63 - a - b - c - d, 'c1': d, 'ber': ber,
                                             'date': datetime.now()})
                            iteration += 1
                            time_elapsed = time.time() - time_start
                            time_average = time_elapsed / iteration
                            print(f"Iteration {iteration}/{iterations}, Lane: {lane}, cm3: {a}, cm2: "
                                f"{b}, cm1: {c}, c0: {63 - a - b - c - d}, c1: {d}, BER: {ber}, Elapsed: "
                                f"{round(time_elapsed, 2)}s, Remaining: "
                                f"{round((time_average * (iterations - iteration)), 2)}s")


def tx_eq_sweep_preset(evb: Shared, lane: int, repetitions: int = 5, pause: float = 0.25, presets:
                       list[str] | None = None) -> None:
    """Contains TX Eq Sweep preset function
    """
    with open(f"tx_eq_sweep_preset_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv", 'w',
              newline='', encoding='UTF-8') as file:
        fields = ['lane', 'preset', 'errors', 'time', 'nbits', 'ber', 'date']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        if presets is None:
            if evb.sp_args['standard_L'] == "100GBASE-KR":
                presets = ['100Gbase_Preset1','100Gbase_Preset2','100Gbase_Preset3','100Gbase_Preset4','100Gbase_Preset5']
            elif evb.sp_args['standard_L'] == "50GBASE-KR":
                presets = ['50Gbase_Preset1','50Gbase_Preset2','50Gbase_Preset3']
            elif standard.lower() == 'pcie1_cc':
                presets = ['p1']
            elif standard.lower() == 'pcie2_cc':
                presets = ['p0', 'p1']
            elif standard.lower() == 'pcie6_cc':
                presets = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10']
            elif standard.lower() == 'pcie3_cc' or standard.lower() == 'pcie4_cc' or standard.lower() == 'pcie5_cc':
                presets = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']
            else:
                presets = ['KR_0dB']

        iterations = len(presets) * repetitions
        time_start = time.time()
        print("\n******** Starting TX EQ Sweep ********")
        for b, preset in enumerate(presets):
            for r in range(repetitions):
                evb.sp_args.update({'tx_eq_L': preset})
                time.sleep(pause)
                evb.initialize()
                ber = evb.meas_ber(lane)
                writer.writerow({'lane': lane, 'preset': preset, 'ber': ber,
                                 'date': datetime.now()})
                iteration = (r + 1) + (b * repetitions)
                time_elapsed = time.time() - time_start
                time_average = time_elapsed / iteration
                print(f"Iteration {iteration}/{iterations}, Lane: {lane}, Preset: {preset}, "
                      f"Elapsed: {round(time_elapsed, 2)}s, Remaining: "
                      f"{round((time_average * (iterations - iteration)), 2)}s")


def tx_eq_sweep_custom(evb: Shared, lane: int, cursors: list[dict[str, int]], repetitions: int = 2,
                       pause: float = 0.25):
    """Contains TX Eq Sweep custom function
    """
    with open(f"tx_eq_sweep_custom_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv", 'w',
              newline='', encoding='UTF-8') as file:
        # soak
        fields = ['lane', 'cm3', 'cm2', 'cm1', 'c0', 'c1', 'errors', 'time', 'nbits', 'ber', 'date']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        iterations = len(cursors) * repetitions
        time_start = time.time()
        print("\n******** Starting TX EQ Sweep ********")
        for i, cursor_dict in enumerate(cursors):
            for r in range(repetitions):
                evb.sp_args.update({'tx_eq_L': 'custom', **{f'{k}_L': v for k, v in cursor_dict.items()}})
                time.sleep(pause)
                evb.initialize()
                ber = evb.meas_ber(lane)
                writer.writerow({'lane': lane, 'cm3': cursor_dict['cm3'], 'cm2': cursor_dict['cm2'],
                                 'cm1': cursor_dict['cm1'], 'c0': cursor_dict['c0'],
                                 'c1': cursor_dict['c1'], 'ber': ber, 'date': datetime.now()})
                iteration = (r + 1) + (i * repetitions)
                time_elapsed = time.time() - time_start
                time_average = time_elapsed / iteration
                print(f"Iteration {iteration}/{iterations}, Lane: {lane}, 'cm3': cursor_dict['cm3'], cm2: {cursor_dict['cm2']}"
                      f", cm1: {cursor_dict['cm1']}, c0: {cursor_dict['c0']}, c1: "
                      f"{cursor_dict['c1']}, BER: {ber}, Elapsed: {round(time_elapsed, 2)}s, "
                      f"Remaining: {round((time_average * (iterations - iteration)), 2)}s")
