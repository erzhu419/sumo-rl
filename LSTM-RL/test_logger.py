from verify_logger import log_event, VerifyLogger
import time

if __name__ == "__main__":
    print("Initializing logger...")
    VerifyLogger("test_logger.log")
    print("Logger initialized.")
    log_event("TEST", "This is a test message", time=123.45)
    print("Log event sent.")
    
    with open("test_logger.log", "r") as f:
        print("Log content:")
        print(f.read())
