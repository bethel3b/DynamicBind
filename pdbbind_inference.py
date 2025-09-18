import logging
import os
import shutil
import subprocess
import time

import pandas as pd
from rdkit import Chem

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# environment variables
relax_python = "/service_data/betty_dev/miniconda3/envs/relax/bin/python"
python = "/service_data/betty_dev/miniconda3/envs/dynamicbind/bin/python"
device = 0


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


def print_summary(
    total_complexes,
    successful_runs,
    failed_runs,
    results_dir,
    summary_path,
    start_time,
    mode,
):
    # Final summary
    total_elapsed = elapsed_time(start_time)
    logger.info("=" * 60)
    logger.info(f"PDBBIND BENCHMARK SUMMARY ({mode.upper()} MODE)")
    logger.info("=" * 60)
    logger.info(f"Total complexes processed: {total_complexes}")
    logger.info(f"Successful runs: {successful_runs}")
    logger.info(f"Failed runs: {failed_runs}")
    logger.info(f"Success rate: {successful_runs / total_complexes * 100:.1f}%")
    logger.info(f"Total processing time: {total_elapsed}")
    logger.info(
        f"Average time per complex: {(time.time() - start_time) / total_complexes:.1f} seconds"
    )
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Summary file: {summary_path}")


def sdf_to_csv(ligand_sdf: str, csv_output_path: str) -> bool:
    """Convert SDF file to CSV with SMILES for DynamicBind input.

    Args:
        ligand_sdf (str): Path to input SDF file
        csv_output_path (str): Path to output CSV file

    Returns:
        bool: Success status
    """
    try:
        # Read SDF file
        suppl = Chem.SDMolSupplier(ligand_sdf)
        smiles_list = []

        # Convert SDF file to CSV
        for mol in suppl:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                smiles_list.append(smiles)

        # Create DataFrame with required 'ligand' column
        df = pd.DataFrame({"ligand": smiles_list})
        df.to_csv(csv_output_path, index=False)
        return True
    except Exception as e:
        logger.error(f"Error converting {ligand_sdf} to CSV: {e}")
        return False


def prepare_complex_files(
    data_dir: str, pdb_id: str, temp_dir: str
) -> tuple[str, str, bool]:
    """Prepare individual protein and ligand files for a specific complex.

    Args:
        data_dir (str): Base data directory containing all complexes
        pdb_id (str): PDB complex ID
        temp_dir (str): Temporary directory for this complex

    Returns:
        tuple[str, str, bool]: (protein_pdb_path, ligand_csv_path, success)
    """
    try:
        # Create temporary directory for this complex
        complex_temp_dir = os.path.join(temp_dir, pdb_id)
        os.makedirs(complex_temp_dir, exist_ok=True)

        # Source file paths
        source_ligand_sdf = os.path.join(
            data_dir,
            pdb_id,
            f"{pdb_id}_ligand.sdf",
        )
        source_protein_pdb = os.path.join(
            data_dir,
            pdb_id,
            f"{pdb_id}_protein.pdb",
        )

        # Destination file paths
        dest_protein_pdb = os.path.join(complex_temp_dir, "protein.pdb")
        dest_ligand_csv = os.path.join(complex_temp_dir, "ligand.csv")

        # Check if source files exist
        if not os.path.exists(source_ligand_sdf):
            logger.warning(f"Warning: Ligand SDF not found for {pdb_id}")
            return None, None, False
        if not os.path.exists(source_protein_pdb):
            logger.warning(f"Warning: Protein PDB not found for {pdb_id}")
            return None, None, False

        # Copy protein file
        shutil.copy2(source_protein_pdb, dest_protein_pdb)

        # Convert ligand SDF to CSV
        if not sdf_to_csv(source_ligand_sdf, dest_ligand_csv):
            logger.error(f"Failed to convert SDF to CSV for {pdb_id}")
            return None, None, False

        return dest_protein_pdb, dest_ligand_csv, True

    except Exception as e:
        logger.error(f"Error preparing files for {pdb_id}: {e}")
        return None, None, False


def run_inference_for_complex(
    data_dir: str,
    pdb_id: str,
    temp_dir: str,
    results_dir: str,
    mode: str = "standard",
    poses: int = None,
    num_samples: int = 1,
) -> bool:
    """Run DynamicBind inference for a single protein-ligand complex.

    Args:
        data_dir (str): Base data directory
        pdb_id (str): PDB complex ID
        temp_dir (str): Temporary directory for processing
        results_dir (str): Directory to store results
        mode (str): Either "hts" or "standard"
        poses (int): Number of poses to generate (overrides mode defaults)
        num_samples (int): Number of samples to generate
    Returns:
        bool: Success status
    """
    logger.info(f"Processing complex {pdb_id} in {mode.upper()} mode...")

    # Prepare files for this complex
    protein_pdb, ligand_csv, prep_success = prepare_complex_files(
        data_dir, pdb_id, temp_dir
    )

    # Check if preparation was successful
    if not prep_success:
        return False

    # Create results directory for this complex
    complex_results_dir = os.path.join(results_dir, pdb_id)
    os.makedirs(complex_results_dir, exist_ok=True)

    # Change to the complex temporary directory for inference
    complex_temp_dir = os.path.join(temp_dir, pdb_id)
    original_cwd = os.getcwd()  # Get the current working directory

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
            inference_steps = 20
            logger.info(
                f"Using HTS mode: {poses or default_poses} poses, {inference_steps} steps"
            )

        else:  # standard mode
            hts_flag = ""
            default_poses = 40
            inference_steps = 20
            logger.info(
                f"Using Standard mode: {poses or default_poses} poses, {inference_steps} steps"
            )

        # Use custom poses if provided, otherwise use mode default
        num_poses = poses if poses is not None else default_poses

        cmd = f"""{env_vars}CUDA_VISIBLE_DEVICES={device} {python} {original_cwd}/run_single_protein_inference.py \\
            protein.pdb ligand.csv \\
            {hts_flag} \\
            --savings_per_complex {num_poses} \\
            --samples_per_complex {num_samples} \\
            --inference_steps {inference_steps} \\
            --paper \\
            --header {pdb_id} \\
            --device {device} \\
            --python {python} \\
            --relax_python {relax_python}"""

        logger.info(f"Running inference for {pdb_id}...")
        result = do(cmd, show=True)

        # Check if results were actually generated
        inference_results = os.path.join(complex_temp_dir, f"results/{pdb_id}")
        affinity_file = os.path.join(inference_results, "affinity_prediction.csv")

        if (
            result == 0
            and os.path.exists(inference_results)
            and os.path.exists(affinity_file)
        ):
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
                logger.info(f"✓ Successfully completed inference for {pdb_id}")
                logger.info(f"Results saved to: {complex_results_dir}")
                # Log number of poses generated
                pose_files = [
                    f
                    for f in os.listdir(complex_results_dir)
                    if f.startswith("rank") and f.endswith(".sdf")
                ]
                logger.info(f"  Generated {len(pose_files)} poses")
                return True
            except Exception as e:
                logger.info(f"✗ Error copying results for {pdb_id}: {e}")
                return False
        else:
            logger.info("*" * 50)
            logger.info(f"✗ Inference failed for {pdb_id}")
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
        logger.info(f"Error during inference for {pdb_id}: {e}")
        return False
    finally:
        os.chdir(original_cwd)


def main(
    mode: str = "standard",
    limit: int | None = None,
    poses: int | None = None,
    num_samples: int = 1,
) -> None:
    start_time = time.time()

    data_dir = "/NAS/data/betty_dev/DynamicBind/betty/pdbbind/data"

    # Set up directories based on mode
    output_dir = "/NAS/project/basilt/evaluation_result/dynamicbind/pdbbind"
    results_dir = os.path.join(output_dir, mode)
    temp_dir = os.path.join(output_dir, "tmp", mode)

    # Create directories
    for directories in [temp_dir, results_dir]:
        os.makedirs(directories, exist_ok=True)

    # Get all complex IDs
    file_name_list = [
        f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))
    ]
    # Apply limit if specified
    if limit is not None:
        file_name_list = file_name_list[:limit]

    total_complexes = len(file_name_list)

    # Determine poses and display info
    if poses is None:
        poses = 3 if mode == "hts" else 40

    logger.info(
        f"Starting PDBbind benchmark with {total_complexes} protein-ligand complexes"
    )
    logger.info(f"Mode: {mode.upper()} ({poses} poses per complex)")
    logger.info(f"Results will be saved to: {results_dir}")
    logger.info("=" * 120 + "\n")

    # Track results
    results_summary = []
    successful_runs = 0
    failed_runs = 0

    for i, pdb_id in enumerate(file_name_list, 1):
        complex_start_time = time.time()
        logger.info(
            "-" * 50 + f" [{i}/{total_complexes}] Processing {pdb_id} " + "-" * 50
        )

        success = run_inference_for_complex(
            data_dir=data_dir,
            pdb_id=pdb_id,
            temp_dir=temp_dir,
            results_dir=results_dir,
            mode=mode,
            poses=poses,
            num_samples=num_samples,
        )

        complex_elapsed = elapsed_time(complex_start_time)

        if success:
            successful_runs += 1
            status = "SUCCESS"
        else:
            failed_runs += 1
            status = "FAILED"

        results_summary.append(
            {
                "complex_id": pdb_id,
                "status": status,
                "processing_time": complex_elapsed,
                "mode": mode,
                "poses": poses,
            }
        )

        logger.info(f"Complex {pdb_id}: {status} (Time: {complex_elapsed})")
        logger.info(
            f"Overall progress: {successful_runs} successful, {failed_runs} failed\n"
        )

        # Clean up temporary files for this complex to save space
        complex_temp_dir = os.path.join(temp_dir, pdb_id)
        if os.path.exists(complex_temp_dir):
            shutil.rmtree(complex_temp_dir)

    # Save comprehensive results summary
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(results_dir, f"benchmark_summary_{mode}.csv")
    summary_df.to_csv(summary_path, index=False)

    # Create consolidated affinity predictions file
    consolidated_results = []
    for pdb_id in file_name_list:
        affinity_file = os.path.join(results_dir, pdb_id, "affinity_prediction.csv")
        if os.path.exists(affinity_file):
            try:
                df_affinity = pd.read_csv(affinity_file)
                df_affinity["complex_id"] = pdb_id
                df_affinity["mode"] = mode
                df_affinity["total_poses"] = poses
                consolidated_results.append(df_affinity)
            except Exception as e:
                logger.warning(
                    f"Warning: Could not read affinity file for {pdb_id}: {e}"
                )

    if consolidated_results:
        consolidated_df = pd.concat(consolidated_results, ignore_index=True)
        consolidated_path = os.path.join(
            results_dir, f"consolidated_affinity_predictions_{mode}.csv"
        )
        consolidated_df.to_csv(consolidated_path, index=False)
        logger.info(f"Consolidated affinity predictions saved to: {consolidated_path}")

    # Clean up temporary directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")

    print_summary(
        total_complexes,
        successful_runs,
        failed_runs,
        results_dir,
        summary_path,
        start_time,
        mode,
    )

    return


if __name__ == "__main__":
    main()
