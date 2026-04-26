import json
import logging
import os
import time

logger = logging.getLogger(__name__)

class MetricsLogger:
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        os.makedirs(os.path.dirname(os.path.abspath(self.log_file_path)), exist_ok=True)
        
        # Initialize or clear the log file
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, "w") as f:
                pass
                
    def log(self, metrics: dict):
        """Append a JSON line with the metrics"""
        metrics["timestamp"] = time.time()
        try:
            with open(self.log_file_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

def check_pause_flag(flag_file_path: str) -> bool:
    """Check if the pause flag file indicates we should pause"""
    if os.path.exists(flag_file_path):
        try:
            with open(flag_file_path, "r") as f:
                data = json.load(f)
                return data.get("pause", False)
        except json.JSONDecodeError:
            pass
    return False

def clear_pause_flag(flag_file_path: str):
    """Clear the pause flag so it doesn't immediately pause on restart"""
    if os.path.exists(flag_file_path):
        try:
            with open(flag_file_path, "w") as f:
                json.dump({"pause": False}, f)
        except Exception as e:
            logger.error(f"Failed to clear pause flag: {e}")

def wait_if_paused(flag_file_path: str, check_interval: float = 2.0):
    """Block execution while the pause flag is set to true"""
    was_paused = False
    while check_pause_flag(flag_file_path):
        if not was_paused:
            logger.info("Training paused. Waiting for resume signal...")
            was_paused = True
        time.sleep(check_interval)
        
    if was_paused:
        logger.info("Resume signal received. Continuing training.")
