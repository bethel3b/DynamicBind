import logging
import os
import shutil
import sys
import time

from tqdm import tqdm

current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
from utils import elapsed_time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    start_time = time.time()
    benchmark_uids = ["P00918", "P07900", "P24941", "P56817", "Q16539"]
    for uid in tqdm(benchmark_uids):
        processed_data_dir = f"/NAS/project/basilt/evaluation_result/dynamicbind/negative_binding/standard/{uid}"

        output_dir = "/NAS/project/MIN-T_benchmark/negative_binding_classification/DynamicBind/predictions"

        # Get files in processed_data_dir if it starts with index
        file_name_list = os.listdir(processed_data_dir)
        file_name_list = [
            file_name for file_name in file_name_list if file_name.startswith("index")
        ]
        logger.info(f"Total files: {len(file_name_list)}")
        logger.info(f"File names: {file_name_list}")

        for file_name in tqdm(file_name_list):
            full_dir = os.path.join(processed_data_dir, file_name)
            unique_id = file_name.split("_")[2]
            file_names_inside_file_name = os.listdir(full_dir)
            required_file_name = [
                file_name
                for file_name in file_names_inside_file_name
                if file_name.endswith("relaxed.sdf")
            ][0]
            required_file_full_path = os.path.join(full_dir, required_file_name)
            shutil.copy2(
                required_file_full_path, os.path.join(output_dir, f"{unique_id}.sdf")
            )

    total_time = elapsed_time(start_time)
    logger.info(
        f"Succesfully Moving processed data to {processed_data_dir} in {total_time}!"
    )
    return


if __name__ == "__main__":
    main()
