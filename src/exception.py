import sys  # Provides access to system-specific parameters and functions (like traceback info)
from src.logger import logging  # Import custom logging setup for capturing errors and information

def error_message_details(error, error_details: sys):
    """
    This function extracts detailed information about an error:
    - Which file the error happened in
    - The exact line number
    - The actual error message
    This allows for better debugging when something goes wrong.
    """
    
    # exc_info() gives traceback information for the most recent exception
    _, _, exc_tb = error_details.exc_info()

    # Get the name of the Python file where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Build the full, descriptive error message
    error_message = (
        "Error occurred in python script name [{0}] "
        "line number [{1}] error message [{2}]"
    ).format(
        file_name,         # File where the error occurred
        exc_tb.tb_lineno,  # Line number of the error
        str(error)         # Actual error message text
    )

    return error_message  # Return the formatted detailed error message


class CustomException(Exception):
    """
    A custom exception class that extends Python's built-in Exception.
    This allows you to create errors with more meaningful and detailed messages.
    """

    def __init__(self, error_message, error_details: sys):
        # Initialize the base Exception class with the original error message
        super().__init__(error_message)

        # Create a detailed error message using our helper function
        self.error_message = error_message_details(
            error_message, 
            error_details=error_details
        )

    def __str__(self):
        """
        When the exception is printed, return the detailed error message.
        """
        return self.error_message


# Example usage (commented out):
# Shows how CustomException wraps the error with full traceback info.
#
# if __name__ == "__main__":
#     try:
#         a = 1 / 0  # This will raise a ZeroDivisionError
#     except Exception as e:
#         logging.info("Dividing by zero error caught.")  # Log the event
#         raise CustomException(e, sys)  # Raise our custom exception for detailed info
