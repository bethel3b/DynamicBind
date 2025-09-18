import logging
import sys


class TeeOutput:
    def __init__(self, filename):
        self.filename = filename
        self.original_stdout = None
        self.original_stderr = None
        self.file = None
        self.log_handler = None

    def __enter__(self):
        # Save original streams
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Open file for writing
        self.file = open(self.filename, "w")

        # Redirect stdout
        sys.stdout = self

        # Add logging handler to capture logger output
        self.log_handler = logging.StreamHandler(self)
        self.log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.log_handler.setFormatter(formatter)

        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove our logging handler
        if self.log_handler:
            root_logger = logging.getLogger()
            root_logger.removeHandler(self.log_handler)

        # Restore original streams
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # Close file
        if self.file:
            self.file.close()

    def write(self, text):
        # Write to both terminal and file
        self.original_stdout.write(text)
        self.original_stdout.flush()

        if self.file and not self.file.closed:
            self.file.write(text)
            self.file.flush()

    def flush(self):
        self.original_stdout.flush()
        if self.file and not self.file.closed:
            self.file.flush()


# Alternative approach: Configure logging to write to both file and console
class DualLogger:
    def __init__(self, filename, logger_name=__name__):
        self.filename = filename
        self.logger_name = logger_name
        self.original_handlers = []

    def __enter__(self):
        # Get the logger
        logger = logging.getLogger(self.logger_name)

        # Save original handlers
        self.original_handlers = logger.handlers.copy()

        # Clear existing handlers
        logger.handlers.clear()

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Create file handler
        file_handler = logging.FileHandler(self.filename, mode="w")
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        return logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original handlers
        logger = logging.getLogger(self.logger_name)
        logger.handlers.clear()
        for handler in self.original_handlers:
            logger.addHandler(handler)
