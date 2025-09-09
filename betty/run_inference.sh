#!/bin/bash
python betty/run_single_protein_inference.py \
    betty/sample_data/1qg8_protein.pdb \
    betty/sample_data/1qg8_ligand.csv \
    --output_dir betty/results \
    --savings_per_complex 40 \
    --inference_steps 20 \
    --header test_0 \
    --device 1 \
    --python /service_data/betty_dev/miniconda3/envs/dynamicbind/bin/python \
    --relax_python /service_data/betty_dev/miniconda3/envs/relax/bin/python 