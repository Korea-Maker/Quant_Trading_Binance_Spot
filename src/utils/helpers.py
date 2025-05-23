# src/utils/helpers.py

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def ensure_dir_exists(directory):
    """디렉토리가 존재하는지 확인하고, 없으면 생성

    Args:
        directory (str): 생성할 디렉토리 경로
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def timestamp_to_datetime(timestamp):
    """타임스탬프를 datetime 객체로 변환

    Args:
        timestamp (int): 밀리초 단위 타임스탬프

    Returns:
        datetime: 변환된 datetime 객체
    """
    return datetime.fromtimestamp(timestamp / 1000)


def interval_to_milliseconds(interval):
    """간격 문자열을 밀리초로 변환

    Args:
        interval (str): 시간 간격 (예: '1m', '1h', '1d')

    Returns:
        int: 밀리초 단위 간격
    """
    # 간격에서 숫자와 단위 분리
    numeric_part = int(''.join(filter(str.isdigit, interval)))
    unit = ''.join(filter(str.isalpha, interval))

    # 단위에 따라 밀리초로 변환
    if unit == 'm':
        return numeric_part * 60 * 1000
    elif unit == 'h':
        return numeric_part * 60 * 60 * 1000
    elif unit == 'd':
        return numeric_part * 24 * 60 * 60 * 1000
    elif unit == 'w':
        return numeric_part * 7 * 24 * 60 * 60 * 1000
    else:
        raise ValueError(f"잘못된 간격 형식: {interval}")


def get_date_range(start_date=None, end_date=None, days=30):
    """날짜 범위 가져오기

    Args:
        start_date (str, optional): 시작일. 형식: 'YYYY-MM-DD'
        end_date (str, optional): 종료일. 형식: 'YYYY-MM-DD'
        days (int, optional): 시작일이 없을 경우 오늘부터 몇 일 전까지 조회할지

    Returns:
        tuple: (시작일, 종료일) datetime 객체
    """
    if end_date is None:
        end_dt = datetime.now()
    else:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    if start_date is None:
        start_dt = end_dt - timedelta(days=days)
    else:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')

    return start_dt, end_dt


def save_to_json(data, filepath):
    """데이터를 JSON 파일로 저장

    Args:
        data (dict): 저장할 데이터
        filepath (str): 파일 경로
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_from_json(filepath):
    """JSON 파일에서 데이터 로드

    Args:
        filepath (str): 파일 경로

    Returns:
        dict: 로드된 데이터
    """
    with open(filepath, 'r') as f:
        return json.load(f)