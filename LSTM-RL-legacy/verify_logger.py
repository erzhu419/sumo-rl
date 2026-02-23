
import logging
import os
import sys

class VerifyLogger:
    _instance = None

    def __new__(cls, log_file="debug_verification.log"):
        if cls._instance is None:
            cls._instance = super(VerifyLogger, cls).__new__(cls)
            cls._instance._initialize(log_file)
        return cls._instance

    def _initialize(self, log_file):
        self.logger = logging.getLogger("VerifyLogger")
        self.logger.setLevel(logging.INFO)
        
        # Avoid adding multiple handlers if initialized multiple times
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file, mode='w')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
            # Also print to stdout for immediate feedback during manual runs, 
            # but maybe less verbose if needed. For now, file is key.
            # console = logging.StreamHandler(sys.stdout)
            # console.setFormatter(formatter)
            # self.logger.addHandler(console)

    def log(self, tag, message, time=None):
        time_str = f"[Time:{time:.2f}]" if time is not None else "[Time:N/A]"
        self.logger.info(f"{tag:<15} {time_str} {message}")

    @classmethod
    def get_logger(cls):
        if cls._instance is None:
            # Default initialization if not called explicitly
            return cls("debug_verification.log")
        return cls._instance

# Global helper for easy import
def log_event(tag, message, time=None):
    VerifyLogger.get_logger().log(tag, message, time)
