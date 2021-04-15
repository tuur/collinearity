#!/bin/sh

source "/home/julius_te/tleeuwenberg/miniconda3/bin/activate"
export PYTHONPATH="/home/julius_te/tleeuwenberg/code/ntcp-collinearity"


PYTHON_SCRIPT="/home/julius_te/tleeuwenberg/code/ntcp-collinearity/exps/compare_methods_citor.py"
PWD=7115
BS=$SLURM_ARRAY_TASK_ID

YAML_PATH="/home/julius_te/tleeuwenberg/yamls/sims/sim-A.yaml" 
python $PYTHON_SCRIPT -yaml $YAML_PATH -pwd $PWD -bs $BS

YAML_PATH="/home/julius_te/tleeuwenberg/yamls/sims/sim-A'.yaml" 
nohup python $PYTHON_SCRIPT -yaml $YAML_PATH -pwd $PWD -bs $BS

YAML_PATH="/home/julius_te/tleeuwenberg/yamls/sims/sim-B.yaml" 
python $PYTHON_SCRIPT -yaml $YAML_PATH -pwd $PWD -bs $BS

YAML_PATH="/home/julius_te/tleeuwenberg/yamls/sims/sim-B'.yaml" 
python $PYTHON_SCRIPT -yaml $YAML_PATH -pwd $PWD -bs $BS

YAML_PATH="/home/julius_te/tleeuwenberg/yamls/sims/sim-C.yaml" 
python $PYTHON_SCRIPT -yaml $YAML_PATH -pwd $PWD -bs $BS

YAML_PATH="/home/julius_te/tleeuwenberg/yamls/sims/sim-C'.yaml" 
nohup python $PYTHON_SCRIPT -yaml $YAML_PATH -pwd $PWD -bs $BS

YAML_PATH="/home/julius_te/tleeuwenberg/yamls/sims/sim-D.yaml" 
python $PYTHON_SCRIPT -yaml $YAML_PATH -pwd $PWD -bs $BS

YAML_PATH="/home/julius_te/tleeuwenberg/yamls/sims/sim-D'.yaml" 
python $PYTHON_SCRIPT -yaml $YAML_PATH -pwd $PWD -bs $BS
