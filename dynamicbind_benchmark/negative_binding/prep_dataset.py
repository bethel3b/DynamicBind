import logging
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
from utils import elapsed_time, extract_unique_uids, save_to_csv_file

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def split_dataset_by_uid(
    df_data: pd.DataFrame, unique_uids: list[str], output_dir: str
) -> None:
    """Split the dataset into chunks based on the unique UIDs.
    Args:
        df_data (pd.DataFrame): Input DataFrame containing the data to process
        unique_uids (list[str]): List of unique UIDs to split the dataset by
        output_dir (str): Directory to save the split datasets

    Returns:
        None
    """
    for uid in tqdm(unique_uids, desc="Splitting dataset based on unique UIDs"):
        output_path = os.path.join(output_dir, f"{uid}.csv")
        df_data_uid = df_data[df_data["uniprot_id"] == uid]
        df_data_uid = df_data_uid.drop(columns=["uniprot_id"])
        df_data_uid = df_data_uid.drop(columns=["ligand_id"])
        df_data_uid = df_data_uid.drop(columns=["id"])
        df_data_uid = df_data_uid.drop(columns=["binding"])
        # remame column name from Cleaned_SMILES to smiles
        df_data_uid = df_data_uid.rename(columns={"Cleaned_SMILES": "ligand"})
        save_to_csv_file(df_processed=df_data_uid, output_path=output_path)
    return


def main(split: bool = True):
    start_time = time.time()
    data_path = "/NAS/project/basilt/evaluation_result/dynamicbind/negative_binding/processed_negative_binding_benchmark.csv"

    output_dir = (
        "/NAS/project/basilt/evaluation_result/dynamicbind/negative_binding/data"
    )

    df_data = pd.read_csv(data_path)

    unique_uids = extract_unique_uids(df_data, "uniprot_id")
    logger.info(unique_uids)
    logger.info(f"Total UIDs: {len(unique_uids):,}")

    if split:
        logger.info("Splitting dataset based on unique UIDs")
        split_dataset_by_uid(df_data, unique_uids, output_dir)

    logger.info(f"Total time: {elapsed_time(start_time)}")


if __name__ == "__main__":
    main()
