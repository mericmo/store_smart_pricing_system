# utils/common.py
import numpy as np
from typing import List, Tuple
import re
import inspect

import pandas as pd


def clean_filename(filename):
    """清理文件名中的非法字符"""
    illegal_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in illegal_chars:
        filename = filename.replace(char, '_')
    filename = re.sub(r'_+', '_', filename)
    filename = filename.strip('_')
    if len(filename) > 100:
        filename = filename[:100]
    return filename

SAVE_FILE_TIMES = 1
def save_to_csv(df: pd.DataFrame, env: str = 'dev') -> bool:
    if df.empty or env != 'dev':
        return False
    caller_frame = inspect.stack()[1]
    caller_method = caller_frame.function
    global SAVE_FILE_TIMES
    df.to_csv(f"temp/{SAVE_FILE_TIMES}_{caller_method}.csv", encoding='utf-8')
    SAVE_FILE_TIMES += 1
    return True
