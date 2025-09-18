import argparse
import logging
import os
import shutil
import sys
import time

import pandas as pd

current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
from utils import do, elapsed_time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# environment variables
relax_python = "/service_data/betty_dev/miniconda3/envs/relax/bin/python"
python = "/service_data/betty_dev/miniconda3/envs/dynamicbind/bin/python"

# DynamicBind root directory - CRITICAL: This must be the DynamicBind repository root
DYNAMICBIND_ROOT = "/NAS/data/betty_dev/DynamicBind"


def run_inference_for_complex(
    ligand_path: str,
    protein_path: str,
    uid: str,
    device: int,
    complex_results_dir: str,
    complex_temp_dir: str,
    mode: str = "standard",
    poses: int = None,
    inference_steps: int = 20,
    num_samples: int = 1,
    num_workers: int = 40,
):
    """Run DynamicBind inference for a single protein-ligand complex."""
    original_cwd = os.getcwd()
    try:
        # Change to the complex temporary directory for inference
        os.chdir(complex_temp_dir)

        # Set environment variables to fix MKL threading conflict
        env_vars = """export MKL_SERVICE_FORCE_INTEL=1; \\
            export MKL_THREADING_LAYER=INTEL; \\
            export OMP_NUM_THREADS=1; \\
            """

        # Configure parameters based on mode
        if mode == "hts":
            hts_flag = "--hts"
            default_poses = 3
            logger.info(
                f"Using HTS mode: {poses or default_poses} poses, {inference_steps} steps"
            )
        else:  # standard mode
            hts_flag = ""
            default_poses = 40
            logger.info(
                f"Using Standard mode: {poses or default_poses} poses, {inference_steps} steps"
            )
        # Use custom poses if provided, otherwise use mode default
        num_poses = poses if poses is not None else default_poses

        cmd = f"""{env_vars}CUDA_VISIBLE_DEVICES={device} {python} \\{original_cwd}/run_single_protein_inference.py \\
            {protein_path} \\
            {ligand_path} \\
            {hts_flag} \\
            --savings_per_complex {num_poses} \\
            --samples_per_complex {num_samples} \\
            --inference_steps {inference_steps} \\
            --paper \\
            --header {uid} \\
            --device {device} \\
            --python {python} \\
            --relax_python {relax_python} \\
            --num_workers {num_workers}
            """

        logger.info(f"Running command from {original_cwd}")
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"Protein: {protein_path}")
        logger.info(f"Ligand: {ligand_path}")
        logger.info(f"Command: {cmd}")
        logger.info(f"Complex results directory: {complex_results_dir}")
        logger.info(f"Complex temp directory: {complex_temp_dir}")

        result = do(cmd, show=True)

        # Check if results were actually generated
        inference_results = os.path.join(complex_temp_dir, f"results/{uid}")
        affinity_file = os.path.join(inference_results, "affinity_prediction.csv")

        if result == 0:
            # Move results to organized results directory
            try:
                for item in os.listdir(inference_results):
                    src = os.path.join(inference_results, item)
                    dst = os.path.join(complex_results_dir, item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)

                logger.info("*" * 50)
                logger.info(f"✓ Successfully completed inference for {uid}")
                logger.info(f"Results saved to: {complex_results_dir}")
                # Log number of poses generated
                total_samples = len(pd.read_csv(ligand_path))
                pose_files = [
                    f for f in os.listdir(complex_results_dir) if f.startswith("index_")
                ]
                logger.info(
                    f"Generated {len(pose_files)} poses / {total_samples} samples"
                )
                return True
            except Exception as e:
                logger.info(f"✗ Error copying results for {uid}: {e}")
                return False
        else:
            logger.info("*" * 50)
            logger.info(f"✗ Inference failed for {uid}")
            logger.info(f"Return code: {result}")
            logger.info(
                f"Results directory exists: {os.path.exists(inference_results)}"
            )
            logger.info(
                f"Affinity file exists: {os.path.exists(affinity_file) if os.path.exists(inference_results) else 'N/A'}"
            )

            # List what files were actually created
            if os.path.exists(complex_temp_dir):
                logger.info(f"Files in temp dir: {os.listdir(complex_temp_dir)}")
                results_dir_path = os.path.join(complex_temp_dir, "results")
                if os.path.exists(results_dir_path):
                    logger.info(f"Files in results: {os.listdir(results_dir_path)}")

            return False

    except Exception as e:
        logger.info(f"Error during inference for {uid}: {e}")
        return False
    finally:
        os.chdir(original_cwd)


def main(
    uid: str = "P00918",
    device: int = 0,
    mode: str = "standard",
    poses: int = None,
):
    start_time = time.time()

    # Input paths - make them absolute
    ligand_path = f"/NAS/project/basilt/evaluation_result/dynamicbind/negative_binding/data/{uid}.csv"
    # ligand_path = "/NAS/data/betty_dev/DynamicBind/dynamicbind_benchmark/negative_binding/data/P00918.csv"
    protein_path = (
        f"/NAS/project/basilt/evaluation_result/dynamicbind/protein_pdb/{uid}.pdb"
    )

    # Output directories
    output_dir = "/NAS/project/basilt/evaluation_result/dynamicbind/negative_binding"
    results_dir = os.path.join(output_dir, "standard_new", uid)
    temp_dir = os.path.join(output_dir, "temp_standard_new", uid)
    for directories in [temp_dir, results_dir]:
        os.makedirs(directories, exist_ok=True)

    logger.info("=" * 120)
    logger.info("Starting negative binding inference")
    logger.info(f"Processing: {uid}")
    logger.info(f"Ligand Path: {ligand_path}")
    logger.info(f"Protein Path: {protein_path}")
    logger.info(f"DynamicBind Root: {DYNAMICBIND_ROOT}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Device: {device}")
    logger.info("=" * 120 + "\n")
    poses = poses if poses is not None else 3 if mode == "hts" else 40

    # Process the dataset
    success = run_inference_for_complex(
        ligand_path=ligand_path,
        protein_path=protein_path,
        uid=uid,
        mode=mode,
        device=device,
        poses=poses,
        num_samples=1,
        complex_results_dir=results_dir,
        complex_temp_dir=temp_dir,
    )

    if success:
        status = "SUCCESS"
    else:
        status = "FAILED"

    logger.info(f"Inference completed: {status}")
    logger.info(f"Total time: {elapsed_time(start_time)} seconds")
    return


def get_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--uid", type=str, default="P00918", required=False)
    argparser.add_argument("--device", type=int, default=0, required=False)
    return argparser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    uid = args.uid
    device = args.device

    main(uid=uid, device=device)
