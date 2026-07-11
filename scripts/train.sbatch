#!/usr/bin/env bash
#SBATCH --job-name=jaundice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/slurm-%j.out
# HPC training launcher. Edit partition/account for your cluster. Real training runs here, not locally.
set -euo pipefail
module load cuda 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true
export PYTHONPATH=src
CONFIG=${1:-configs/base.yaml}
TAG=${2:-v1_all}
python -m jaundice.train.train --config "$CONFIG" --tag "$TAG"
