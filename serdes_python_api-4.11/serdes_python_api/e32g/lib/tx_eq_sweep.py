"""Contains various tx eq sweep functions
"""

import csv
import time
import math
from datetime import datetime

from serdes_python_api.e32g.e32g import E32G

REVISION = '$Revision: #3 $'
DATE = '$Date: 2024/09/30 $'

def tx_eq_sweep(evb: E32G,
                lane: int,
                repetitions: int = 2,
                pause: float = 0.25,
                pre_lo: int = 0,
                pre_hi: int = 63,
                pre_step: int = 4,
                main_lo: int = 0,
                main_hi: int = 24,
                main_step: int = 1,
                post_lo: int = 0,
                post_hi: int = 63,
                post_step: int = 4,) -> None:

    """Sweeps all cursors

    :param repetitions: Number of configurations at each cursor combination, defaults to 2
    :param pause: Pause time (seconds) between configurations, defaults to 0.25
    """

    with open(f"tx_eq_sweep_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv", 'w', newline='',
              encoding='UTF-8') as file:
        fields = ['lane', 'pre', 'main', 'post', 'ber', 'date']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        iteration = 0
        print("\n******** Starting TX EQ Sweep ********")
        for pre in range(pre_lo, pre_hi + 1, pre_step):
            for post in range(post_lo, post_hi + 1, post_step):
                for main in range(main_lo, main_hi+ 1, main_step):
                    if main < 0 :
                        continue
                    for _ in range(repetitions):
                        evb.sp_args.update({'tx_eq_L': 'custom', 'pre_L': pre, 'post_L': post,
                                            'main_L': main})

                        time.sleep(pause)
                        evb.set_tx_eq(lane,lane,pre,post,main)
                        ber = evb.meas_ber(lane, math.pow(10,-9.5))

                        writer.writerow({'lane': lane, 'pre': pre, 'post': post, 'main': main,
                                            'ber': ber, 'date': datetime.now()})

                        iteration += 1
                        print(f"Iteration {iteration}, Lane: {lane}, pre: {pre}, post: {post}, "
                                f"main: {main}, BER: {ber}")

def tx_eq_sweep_fixed(evb: E32G,
                      lane: int,
                      repetitions: int = 2,
                      pause: float = 0.25,
                      pre_lo: int = 0,
                      pre_hi: int = 4,
                      pre_step: int = 1,
                      post_lo: int = 0,
                      post_hi: int = 1,
                      post_step: int = 1) -> None:
    """Sweeps all cursors, if sum is less than 24 the extra gets added to main

    :param repetitions: Number of configurations at each cursor combination, defaults to 2
    :param pause: Pause time (seconds) between configurations, defaults to 0.25
    """
    with open(f"tx_eq_sweep_fixed_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv", 'w', newline='',
              encoding='UTF-8') as file:
        fields = ['lane', 'pre', 'main', 'post', 'ber', 'date']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        iteration = 0
        iterations = int((pre_hi - pre_lo + 1) / pre_step) * int((post_hi - post_lo + 1) / post_step) * repetitions
        time_start = time.time()
        print("\n******** Starting TX EQ Sweep Fixed ********")

        for pre in range(pre_lo, pre_hi + 1, pre_step):
            for post in range(post_lo, post_hi + 1, post_step):
                main = 24 - (math.ceil(pre/4) + math.ceil(post/4))
                if main < 0:
                    continue
                for _ in range(repetitions):
                    evb.sp_args.update({'tx_eq_L': 'custom', 'pre_L': pre, 'post_L': post,
                                        'main_L': main})

                    time.sleep(pause)
                    evb.set_tx_eq(lane,lane,pre,post,main)
                    ber = evb.meas_ber(lane, math.pow(10,-9.5))

                    writer.writerow({'lane': lane, 'pre': pre, 'post': post, 'main': main,
                                        'ber': ber, 'date': datetime.now()})

                    iteration += 1
                    time_elapsed = time.time() - time_start
                    time_average = time_elapsed / iteration
                    print(f"Iteration {iteration}/{iterations}, Lane: {lane}, pre: {pre}, post: "
                            f"{post}, main: {main}, BER: {ber}, Elapsed: "
                            f"{round(time_elapsed, 2)}s, Remaining: "
                            f"{round((time_average * (iterations - iteration)), 2)}s")


def tx_eq_sweep_preset(evb: E32G, lane: int, presets: list[str] | None = None, repetitions: int = 2, pause: float = 0.25,) -> None:
    """Sweep cursors over presets

    :param repetitions: Number of configurations at each cursor combination, defaults to 2
    :param pause: Pause time (seconds) between configurations, defaults to 0.25
    :param presets: TX EQ preset, defaults to None
    """

    if presets is None:
        presets = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']

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
                evb.sp_args.update({'tx_eq_pre_L': None})
                evb.sp_args.update({'tx_eq_post_L': None})
                evb.sp_args.update({'tx_eq_main_L': None})
                # Configure Part
                evb.initialize()
                evb.sync_pm(lane)
                evb.get_polarity(tx_lane=lane, rx_lane=lane)
                ber = evb.meas_ber(lane, math.pow(10,-9.5))
                writer.writerow({'lane': lane, 'preset': preset,
                                'ber': ber, 'date': datetime.now()})

                iteration = (r + 1) + (i * repetitions)
                time_elapsed = time.time() - time_start
                time_average = time_elapsed / iteration
                print(f"Iteration {iteration}/{iterations}, Lane: {lane}, Preset: {preset}, "
                      f"Elapsed: {round(time_elapsed, 2)}s, Remaining: "
                      f"{round((time_average * (iterations - iteration)), 2)}s")


def tx_eq_sweep_custom(evb: E32G, lane: int, cursors: list[dict], repetitions: int = 2,
                       pause: float = 0.25):
    """Sweep cursors over custom values

    :param cursors: list of dictionaries with k = cursor (pre, post, main), v = value
    :param repetitions: Number of configurations at each cursor combination, defaults to 2
    :param pause: Pause time (seconds) between configurations, defaults to 0.25
    """

    with open(f"tx_eq_sweep_custom_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv", 'w',
              newline='', encoding='UTF-8') as file:
        fields = ['lane', 'pre', 'main', 'post','ber', 'date']
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
                main = 24 - (math.ceil(cursor_dict['pre']/4) + math.ceil(cursor_dict['post']/4))
                if(cursor_dict['main'] > main):
                    print(f"Eq-Pre Setting : pre: {cursor_dict['pre']}"
                      f", post: {cursor_dict['post']}, main: {cursor_dict['main']} is not valid")
                    continue
                evb.set_tx_eq(lane,lane,cursor_dict['pre'],cursor_dict['post'],cursor_dict['main'])
                ber = evb.meas_ber(lane, math.pow(10,-9.5))
                writer.writerow({'lane': lane, 'pre': cursor_dict['pre'],
                                 'post': cursor_dict['post'], 'main': cursor_dict['main'],
                                 'ber': ber, 'date': datetime.now()})

                iteration = (r + 1) + (i * repetitions)
                time_elapsed = time.time() - time_start
                time_average = time_elapsed / iteration
                print(f"Iteration {iteration}/{iterations}, Lane: {lane}, pre: {cursor_dict['pre']}"
                      f", post: {cursor_dict['post']}, main: {cursor_dict['main']},"
                      f"BER: {ber}, Elapsed: {round(time_elapsed, 2)}s, "
                      f"Remaining: {round((time_average * (iterations - iteration)), 2)}s")
