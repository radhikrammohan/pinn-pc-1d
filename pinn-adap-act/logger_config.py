import logging
import os
import csv

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    # Stream handler for console output (optional)
    # stream = logging.StreamHandler()
    # stream.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    # logger.addHandler(stream)
    
    return logger