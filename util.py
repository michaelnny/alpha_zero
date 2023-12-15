# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Mapping, Text, Any
import logging
import sys
import time
import timeit
from collections import deque


def get_time_stamp(file_name: bool = False) -> str:
    t = time.localtime()
    if file_name:
        return time.strftime('%Y%m%d_%H%M%S', t)
    else:
        return time.strftime('%Y-%m-%d %H:%M:%S', t)


def extract_args_from_flags_dict(flags_dict: Mapping[Text, Any]) -> Mapping[Text, Any]:
    # Default arguments from the absl flags
    keys_to_exclude = [
        'logtostderr',
        'alsologtostderr',
        'log_dir',
        'v',
        'verbosity',
        'logger_levels',
        'stderrthreshold',
        'showprefixforinfo',
        'run_with_pdb',
        'pdb_post_mortem',
        'pdb',
        'run_with_profiling',
        'profile_file',
        'use_cprofile_for_profiling',
        'only_check_args',
        '?',
        'help',
        'helpshort',
        'helpfull',
        'helpxml',
    ]

    args = {}

    for k, v in flags_dict.items():
        if k not in keys_to_exclude:
            args[k] = v

    return args


def create_logger(level='INFO'):
    handler = logging.StreamHandler(stream=sys.stderr)
    formatter = logging.Formatter(
        fmt='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    veb = logging.INFO
    level = str(level).upper()
    if level == 'DEBUG':
        veb = logging.DEBUG
    logger.setLevel(veb)
    logger.addHandler(handler)

    return logger


class Timer:
    def __init__(self, max_history=100):
        self.history = deque(maxlen=max_history)

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.history.append(timeit.default_timer() - self.start)

    def mean_time(self):
        if len(self.history) == 0:
            return 0

        return sum(self.history) / len(self.history)

    def last_time(self):
        if len(self.history) == 0:
            return 0

        return self.history[-1]
