import logging  # Importing the logging module for logging messages during runtime
import os  # Importing the os module to interact with the operating system, like creating directories or working with file paths
from datetime import datetime  # Importing datetime module to get the current date and time

# Generate a log file name based on the current date and time (e.g., '2025-12-12_14-30-45.log')
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"  # Format the current date and time to use as the log file name

# Set the path where the log files will be stored: current working directory + "logs" + log file name
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)  # Creates the full path for storing logs, e.g., "/current/directory/logs/2025-12-12_14-30-45.log"

# Create the "logs" directory if it does not exist (exist_ok=True ensures no error if the directory is already present)
os.makedirs(logs_path, exist_ok=True)  # Ensure the "logs" directory exists

# Set the final path where the log file will be stored
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)  # Combine the logs directory path with the log file name to form the full log file path

# Configure logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Set the log file path where logs will be written
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Define the format of log entries: time, line number, module, log level, message
    level=logging.INFO  # Set the logging level to INFO (INFO, WARNING, ERROR, CRITICAL messages will be logged)
)

# Main block: Only runs if this script is executed directly (not imported)
if __name__ == "__main__":
    logging.info("Logging has started.")  # Log an informational message to indicate that logging has started
