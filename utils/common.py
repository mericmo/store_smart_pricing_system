# utils/common.py
import numpy as np
from typing import List, Tuple
import re
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