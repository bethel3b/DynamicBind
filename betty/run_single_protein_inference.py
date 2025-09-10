#!/home/zhangjx/anaconda3/envs/dynamicbind/bin/python
import logging
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd
import rdkit.Chem as Chem

# Default values
HEADER = "test"
PYTHON = "/service_data/betty_dev/miniconda3/envs/dynamicbind/bin/python"
RELAX_PYTHON = "/service_data/betty_dev/miniconda3/envs/relax/bin/python"
PROTEIN_FILE = "betty/sample_data/1qg8_protein.pdb"
LIGAND_FILE = "betty/sample_data/1qg8_ligand.csv"
OUTPUT_DIR = "betty/results"
RESULT_DIR = "inference_results"
SAMPLES_PER_COMPLEX = 10
SAVINGS_PER_COMPLEX = 10
INFERENCE_STEPS = 20
NUM_WORKERS = 20
MODEL = 1
SEED = 42
RIGID_PROTEIN = False
HTS = False
DEVICE = 0
NO_INFERENCE = False
NO_RELAX = False
MOVIE = False
PROTEIN_PATH_IN_LIGAND_FILE = False
NO_CLEAN = False
LIGAND_IS_SDF = False
PAPER = False


# -------------------------------------------------------------------
# Set up function to run commands
# -------------------------------------------------------------------
def do(cmd, get=False, show=True):
    if get:
        out = (
            subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            .communicate()[0]
            .decode()
        )
        if show:
            print(out, end="")
        return out
    else:
        return subprocess.Popen(cmd, shell=True).wait()


import argparse


def get_arguments():
    parser = argparse.ArgumentParser(
        description="python run_single_protein_inference.py betty/sample_data/1qg8_protein.pdb betty/sample_data/1qg8_ligand.csv --header test_0"
    )

    parser.add_argument(
        "--proteinFile",
        type=str,
        default=PROTEIN_FILE,
        help="protein file",
    )
    parser.add_argument(
        "--ligandFile",
        type=str,
        default=LIGAND_FILE,
        help="contians the smiles, should contain a column named ligand",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="informative name used to name result folder",
    )
    parser.add_argument(
        "--samples_per_complex",
        type=int,
        default=SAMPLES_PER_COMPLEX,
        help="num of samples data generated.",
    )
    parser.add_argument(
        "--savings_per_complex",
        type=int,
        default=SAVINGS_PER_COMPLEX,
        help="num of samples data saved for movie generation.",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=INFERENCE_STEPS,
        help="num of coordinate updates. (movie frames)",
    )
    parser.add_argument(
        "--header",
        type=str,
        default=HEADER,
        help="informative name used to name result folder",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=RESULT_DIR,
        help="result folder.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=DEVICE,
        help="CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--no_inference",
        action="store_true",
        default=NO_INFERENCE,
        help="used, when the inference part is already done.",
    )
    parser.add_argument(
        "--no_relax",
        action="store_true",
        default=NO_RELAX,
        help="by default, the last frame will be relaxed.",
    )
    parser.add_argument(
        "--movie",
        action="store_true",
        default=MOVIE,
        help="by default, no movie will generated.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=PYTHON,
        help="point to the python in dynamicbind env.",
    )
    parser.add_argument(
        "--relax_python",
        type=str,
        default=RELAX_PYTHON,
        help="point to the python in relax env.",
    )
    parser.add_argument(
        "-l",
        "--protein_path_in_ligandFile",
        action="store_true",
        default=PROTEIN_PATH_IN_LIGAND_FILE,
        help="read the protein from the protein_path in ligandFile.",
    )
    parser.add_argument(
        "--no_clean",
        action="store_true",
        default=NO_CLEAN,
        help="by default, the input protein file will be cleaned. only take effect, when protein_path_in_ligandFile is true",
    )
    parser.add_argument(
        "-s",
        "--ligand_is_sdf",
        action="store_true",
        default=LIGAND_IS_SDF,
        help="ligand file is in sdf format.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of workers for relaxing final step structure",
    )
    parser.add_argument(
        "-p",
        "--paper",
        action="store_true",
        default=PAPER,
        help="use paper version model.",
    )
    parser.add_argument(
        "--model",
        type=int,
        default=MODEL,
        help="default model version",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="set seed number",
    )
    parser.add_argument(
        "--rigid_protein",
        action="store_true",
        default=RIGID_PROTEIN,
        help="Use no noise in the final step of the reverse diffusion",
    )
    parser.add_argument(
        "--hts",
        action="store_true",
        default=HTS,
        help="high-throughput mode",
    )

    args = parser.parse_args()
    return args


args = get_arguments()


# -------------------------------------------------------------------
# Set up output directory
# -------------------------------------------------------------------
output_dir = args.output_dir

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
output_dir = f"{output_dir}_{timestamp}"
log_path = os.path.join(output_dir, "logs", "run.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

# -------------------------------------------------------------------
# Set up logging
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
handler = logging.FileHandler(log_path)
logger = logging.getLogger("")
logger.addHandler(handler)

logging.info(f"""\
{" ".join(sys.argv)}
{timestamp}
--------------------------------
""")

# -------------------------------------------------------------------
# Set up environment
# -------------------------------------------------------------------
python = args.python
relax_python = args.relax_python

os.environ["PATH"] = os.path.dirname(relax_python) + ":" + os.environ["PATH"]
file_path = os.path.realpath(__file__)
script_folder = os.path.dirname(file_path)
parent_script_folder = os.path.dirname(script_folder)
print(f"Run directory: {file_path},\nParent script folder: {parent_script_folder}")

# -------------------------------------------------------------------
# Set up ligand file and protein file
# -------------------------------------------------------------------
protein_path_in_ligandFile = args.protein_path_in_ligandFile
no_clean = args.no_clean
ligand_is_sdf = args.ligand_is_sdf
ligandFile = args.ligandFile
proteinFile = args.proteinFile

processed_data_dir = os.path.join(output_dir, "processed_data")
os.makedirs(processed_data_dir, exist_ok=True)

clean_pdb_script_path = os.path.join(parent_script_folder, "clean_pdb.py")
if protein_path_in_ligandFile:  # default is False
    if no_clean:  # default is False
        ligandFile_with_protein_path = ligandFile
    else:
        # Path to ligand file with protein path
        ligandFile_with_protein_path = os.path.join(
            processed_data_dir, "ligandFile_with_protein_path.csv"
        )
        cmd = f"{relax_python} {clean_pdb_script_path} \
            {ligandFile} \
            {ligandFile_with_protein_path}"
        do(cmd)
    ligands = pd.read_csv(ligandFile_with_protein_path)
    assert "ligand" in ligands.columns
    assert "protein_path" in ligands.columns

elif ligand_is_sdf:  # default is False
    # Path to cleaned protein file
    cleaned_proteinFile = os.path.join(
        processed_data_dir,
        "cleaned_input_proteinFile.pdb",
    )
    # Path to ligand file with protein path
    ligandFile_with_protein_path = os.path.join(
        processed_data_dir,
        "ligandFile_with_protein_path.csv",
    )
    # Clean protein file
    cmd = f"{relax_python} {clean_pdb_script_path} \
        {proteinFile} \
        {cleaned_proteinFile}"
    do(cmd)

    # reorder the mol atom number as in smiles.
    ligandFile = f"{output_dir}/" + os.path.basename(ligandFile)
    mol = Chem.MolFromMolFile(ligandFile)
    _ = Chem.MolToSmiles(mol)
    m_order = list(
        mol.GetPropsAsDict(includePrivate=True, includeComputed=True)[
            "_smilesAtomOutputOrder"
        ]
    )
    mol = Chem.RenumberAtoms(mol, m_order)
    w = Chem.SDWriter(ligandFile)
    w.write(mol)
    w.close()
    ligands = pd.DataFrame(
        {"ligand": [ligandFile], "protein_path": [cleaned_proteinFile]}
    )
    ligands.to_csv(ligandFile_with_protein_path, index=0)

else:  # this is the default case - protein file is in csv format
    # Path to cleaned protein file
    cleaned_proteinFile = os.path.join(
        processed_data_dir, "cleaned_input_proteinFile.pdb"
    )
    # Path to ligand file with protein path
    ligandFile_with_protein_path = os.path.join(
        processed_data_dir, "ligandFile_with_protein_path.csv"
    )
    # Clean protein file
    cmd = f"{relax_python} {clean_pdb_script_path} \
        {proteinFile} \
        {cleaned_proteinFile}"
    do(cmd)

    # Read ligand file and add protein path to ligand file columns and save
    ligands = pd.read_csv(ligandFile)
    assert "ligand" in ligands.columns
    ligands["protein_path"] = cleaned_proteinFile
    ligands.to_csv(ligandFile_with_protein_path, index=0)

# -------------------------------------------------------------------
# Set up model workdir and checkpoint
# -------------------------------------------------------------------
header = args.header
paper = args.paper
model = args.model

model_workdir = os.path.join(
    script_folder, "workdir", "big_score_model_sanyueqi_with_time"
)
if paper:  # default is False
    ckpt = "ema_inference_epoch314_model.pt"
else:  # this is the default case
    if model == 1:  # default is 1
        ckpt = "pro_ema_inference_epoch138_model.pt"

# -------------------------------------------------------------------
# Set up protein dynamic - no noise in the final step of the reverse diffusion
# -------------------------------------------------------------------
rigid_protein = args.rigid_protein

if not rigid_protein:  # default is False
    protein_dynamic = "--protein_dynamic"
else:  # this is the default case
    protein_dynamic = ""

# -------------------------------------------------------------------
# Set up arguments
# -------------------------------------------------------------------
hts = args.hts
device = args.device
seed = args.seed
inference_steps = args.inference_steps
samples_per_complex = args.samples_per_complex
savings_per_complex = args.savings_per_complex
results = args.results
no_inference = args.no_inference
no_relax = args.no_relax
movie = args.movie

num_workers = args.num_workers

# -------------------------------------------------------------------
# Set up esm2 output directory and model directory
# -------------------------------------------------------------------
esm2_output_dir = os.path.join(output_dir, "esm2_output")
os.makedirs(esm2_output_dir, exist_ok=True)
model_dir = os.path.join(parent_script_folder, "esm_models")
esm_embedding_preparation_script_path = os.path.join(
    parent_script_folder, "datasets", "esm_embedding_preparation.py"
)
esm_festa_output_path = os.path.join(
    processed_data_dir, f"prepared_for_esm_{header}.fasta"
)
esm_embedding_extraction_script_path = os.path.join(
    parent_script_folder, "esm", "scripts", "extract.py"
)

# -------------------------------------------------------------------
# Set up hts mode - generate esm embeddings for all ligands
# -------------------------------------------------------------------
if hts:  # default is False
    # Run esm embedding preparation
    cmd = f"{python} {esm_embedding_preparation_script_path} \
        --protein_ligand_csv {ligandFile_with_protein_path} \
        --out_file {esm_festa_output_path}"
    if protein_path_in_ligandFile:
        cmd += f" --protein_path {proteinFile}"
    do(cmd)

    # Run esm embedding extraction
    cmd = f"CUDA_VISIBLE_DEVICES={device} \
        {python} {esm_embedding_extraction_script_path} \
        esm2_t33_650M_UR50D \
        {esm_festa_output_path} \
        {esm2_output_dir} \
        --repr_layers 33 \
        --include per_tok \
        --truncation_seq_length 10000 \
        --model_dir {model_dir}"
    do(cmd)

    # Run screening
    screening_script_path = os.path.join(parent_script_folder, "screening.py")
    screening_output_dir = os.path.join(output_dir, results, header)
    cmd = f"CUDA_VISIBLE_DEVICES={device} \
        {python} {screening_script_path} \
        --seed {seed} \
        --ckpt {ckpt} \
        {protein_dynamic}"
    cmd += f" --save_visualisation \
        --model_dir {model_workdir}  \
        --protein_ligand_csv {ligandFile_with_protein_path} "
    cmd += f" --esm_embeddings_path {esm2_output_dir} \
        --out_dir {screening_output_dir} \
        --inference_steps {inference_steps} \
        --samples_per_complex {samples_per_complex} \
        --savings_per_complex {savings_per_complex} \
        --batch_size 5 \
        --actual_steps {inference_steps} \
        --no_final_step_noise"
    do(cmd)
    print("hts complete.")

# -------------------------------------------------------------------
# Set up non-hts mode - generate esm embeddings for each ligand
# -------------------------------------------------------------------
else:
    # -------------------------------------------------------------------
    # Set up inference
    # -------------------------------------------------------------------
    if not no_inference:  # default is False - so True
        # Run esm embedding preparation
        cmd = f"{python} {esm_embedding_preparation_script_path} \
            --protein_ligand_csv {ligandFile_with_protein_path} \
            --out_file {esm_festa_output_path}"
        do(cmd)

        # Run esm embedding extraction
        cmd = f"CUDA_VISIBLE_DEVICES={device} \
            {python} {esm_embedding_extraction_script_path} \
            esm2_t33_650M_UR50D \
            {esm_festa_output_path} \
            {esm2_output_dir} \
            --repr_layers 33 \
            --include per_tok \
            --truncation_seq_length 10000 \
            --model_dir {model_dir}"
        do(cmd)

        # Run inference
        inference_script_path = os.path.join(parent_script_folder, "inference.py")
        inference_output_dir = os.path.join(output_dir, results, header)
        cmd = f"CUDA_VISIBLE_DEVICES={device} \
            {python} {inference_script_path} \
            --seed {seed} \
            --ckpt {ckpt} \
            {protein_dynamic}"
        cmd += f" --save_visualisation \
            --model_dir {model_workdir}  \
            --protein_ligand_csv {ligandFile_with_protein_path} "
        cmd += f" --esm_embeddings_path {esm2_output_dir} \
            --out_dir {inference_output_dir} \
            --inference_steps {inference_steps} \
            --samples_per_complex {samples_per_complex} \
            --savings_per_complex {savings_per_complex} \
            --batch_size 5 \
            --actual_steps {inference_steps} \
            --no_final_step_noise"
        do(cmd)
        print("inference complete.")

    relax_final_script_path = os.path.join(parent_script_folder, "relax_final.py")
    relax_final_output_dir = os.path.join(output_dir, results, header)
    # -------------------------------------------------------------------
    # Set up relax final step structure
    # -------------------------------------------------------------------
    if not no_relax:  # default is False - so do
        cmd = f"CUDA_VISIBLE_DEVICES={device} \
            {relax_python} \
            {relax_final_script_path} \
            --results_path {relax_final_output_dir} \
            --samples_per_complex {samples_per_complex} \
            --num_workers {num_workers}"
        # print("relax final step structure.")
        # exit()
        do(cmd)
        print("final step structure relax complete.")
    # -------------------------------------------------------------------
    # Set up movie generation
    # -------------------------------------------------------------------s
    if movie:  # default is False
        movie_generation_script_path = os.path.join(
            parent_script_folder, "movie_generation.py"
        )
        for i in range(len(ligands)):
            movie_output_dir = os.path.join(
                output_dir, results, header, f"index{i}_idx_{i}"
            )
            cmd = f"CUDA_VISIBLE_DEVICES={device} \
                {relax_python} {movie_generation_script_path} \
                {movie_output_dir} \
                --python  {python} \
                --relax_python {relax_python} \
                --inference_steps {inference_steps}"
            do(cmd)
            print(cmd)
