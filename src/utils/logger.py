# src/utils/logger.py

import os
import logging
from datetime import datetime
from src.config.settings import LOG_DIR, LOG_LEVEL, APP_MODE

def get_logger(name):
    """로거 생성 함수

    Args:
        name (str): 로거 이름

    Returns:
        logging.Logger: 설정된 로거 객체
    """
    # 로거 생성
    logger = logging.getLogger(name)

    # 로그 레벨 설정
    log_level = getattr(logging, LOG_LEVEL)
    logger.setLevel(log_level)

    # 이미 핸들러가 설정되어 있으면 반환
    if logger.handlers:
        return logger

    # 로그 파일명 설정
    today = datetime.now().strftime('%Y-%m-%d')
    log_file = os.path.join(LOG_DIR, f"{today}_{name.replace('.', '_')}.log")

    # 파일 핸들러
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # 포맷터 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 개발 모드에서는 자세한 로그
    if APP_MODE == 'development':
        logger.setLevel(logging.DEBUG)

    return logger

# Test
# if __name__ == "__main__":
#     logger = get_logger("test_test_logger")
#     logger.info("Test_Message")