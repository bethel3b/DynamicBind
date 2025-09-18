import logging
import os
import subprocess
import time

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_column(df_data: pd.DataFrame, target_column: str) -> None:
    """Validate that a specified column exists in the DataFrame.

    Args:
        df_data (pd.DataFrame): DataFrame to validate against
        target_column (str): Name of the column to check for existence

    Returns:
        None: This function returns nothing if validation passes

    Raises:
        ValueError: If the target_column does not exist in the DataFrame's columnss
    """
    if target_column not in df_data.columns:
        raise ValueError(f"Column '{target_column}' not found in dataset")


def extract_unique_uids(
    df_data: pd.DataFrame, target_column: str, verbose: bool = True
) -> list[str]:
    """Extract unique UIDs from a specified column in a DataFrame.
    Args:
        df_data (pd.DataFrame): Input DataFrame containing the data to process
        target_column (str):  Name of the column containing UIDs to extract
        verbose (bool): If True, displays detailed statistics.
            Defaults to True.

    Returns:
        list[str]: List of unique UIDs found in the target column

    Raises:
        ValueError: If the target_column does not exist in the DataFrame's columns
    """
    validate_column(df_data=df_data, target_column=target_column)
    uid_series = df_data[target_column]
    unique_uids = uid_series.unique().tolist()

    if verbose:
        logger.info(f"Total UIDs: {len(uid_series):,}")
        logger.info(f"Unique UIDs: {len(unique_uids):,}")
    return unique_uids


def save_to_csv_file(df_processed: pd.DataFrame, output_path: str) -> None:
    """Save processed data to a csv file.


    Args:
        df_processed (pd.DataFrame): dataframe
        output_path (str): Full path where the processed file should be saved

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    df_processed.to_csv(output_path, index=False)

    logger.info(f"Successfully saved {len(df_processed)} data to {output_path}\n")
    return


def elapsed_time(start_time: float) -> str:
    """Calculate and format elapsed time.

    Args:
        start_time(float): Time when processing started (from time.time())

    Returns:
        str: Formatted time string (HH:MM:SS)
    """
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def do(cmd: str, get: bool = False, show: bool = True) -> int:
    """Run a command and return the output.

    Args:
        cmd (str): The command to run
        get (bool): Whether to return the output
        show (bool): Whether to print the output
    """
    if get:
        out = (
            subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            .communicate()[0]
            .decode()
        )
        if show:
            logger.info(out, end="")
        return out
    else:
        return subprocess.Popen(cmd, shell=True).wait()
