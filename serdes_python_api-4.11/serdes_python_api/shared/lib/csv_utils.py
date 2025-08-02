"""Contains functions for interacting with csv files
"""

import csv
import os
from datetime import datetime

REVISION = '$Revision: #1 $'
DATE = '$Date: 2024/01/04 $'


def append_dict_to_csv(fname: str, to_append: dict, add_timestamp: bool = True,
                       test_num: int | None = None) -> None:
    """Keys, including provided timestamp and test number arguments, must be maintained in
    subsequent tests
    """

    if add_timestamp:
        to_append = {'Date': datetime.now().isoformat(sep=' ', timespec='seconds'), **to_append}
    if test_num:
        to_append = {'Test Number': test_num, **to_append}
    with open(fname, 'a+', newline='', encoding='utf-8') as f:
        size = os.path.getsize(fname)
        writer = csv.DictWriter(f, fieldnames=to_append.keys())
        if not size:
            writer.writeheader()
        writer.writerow(to_append)


def append_list_to_csv(fname: str, to_append: list, headers: list[str] | None = None,
                       add_timestamp: bool = True, test_num: int | None = None) -> None:
    """Headers, timestamp, and test_num arguments must be maintained in subsequent tests
    """

    if not headers:
        headers = [f"Var {i}" for i in range(len(to_append))]
    if add_timestamp:
        headers = ["Date"] + headers
        to_append = [datetime.now().isoformat(sep=' ', timespec='seconds')] + to_append
    if test_num:
        headers = ["Test Number"] + headers
        to_append = [test_num] + to_append
    with open(fname, 'a+', newline='', encoding='utf-8') as f:
        size = os.path.getsize(fname)
        writer = csv.writer(f)
        if not size:
            writer.writerow(headers)
        writer.writerow(to_append)
