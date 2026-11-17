#!/bin/bash

#SBATCH --job-name=anomalies-rl
#SBATCH --output=/home/kumarj/links/scratch/anomalies_%j.out
#SBATCH --error=/home/kumarj/links/scratch/anomalies_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=def-london


# ============================================================
# START
# ============================================================

start_time=$(date +%s)

echo "=========================================="
echo "ANOMALIES RL TRAINING - CPU"
echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Job started:   $(date)"


# ============================================================
# MODULES
# ============================================================

module purge
module load StdEnv/2023
module load gcc/12.3
module load arrow
module load python/3.10


# ============================================================
# PATHS
# ============================================================

CODE_DIR="/project/def-london/kumarj/gitrepos/smeft-anomaly/reinforcesmeft"
WORK_DIR="/home/kumarj/links/scratch/anomalies"

mkdir -p "$WORK_DIR"

echo ""
echo "Code directory:"
echo "$CODE_DIR"

echo ""
echo "Working/output directory:"
echo "$WORK_DIR"


# ============================================================
# PYTHON ENVIRONMENT
# ============================================================

source /project/def-london/kumarj/gitrepos/smeft-anomaly/ENV/bin/activate

echo ""
echo "Python:"
which python
python --version


# ============================================================
# CPU THREADING
# ============================================================

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1

echo ""
echo "OMP_NUM_THREADS      = $OMP_NUM_THREADS"
echo "MKL_NUM_THREADS      = $MKL_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS = $OPENBLAS_NUM_THREADS"


# ============================================================
# CHECK PACKAGE
# ============================================================

echo ""
echo "Checking anomalies package..."

python -c "import reinforcesmeft; print('reinforcesmeft:', reinforcesmeft.__file__)"

if [ $? -ne 0 ]; then
    echo "ERROR: anomalies could not be imported."
    exit 1
fi


# ============================================================
# RUN FROM WRITABLE SCRATCH DIRECTORY
# ============================================================

cd "$WORK_DIR" || exit 1

echo ""
echo "Current working directory:"
pwd

echo ""
echo "Starting RL analysis..."
echo ""


# IMPORTANT:
# Exactly ONE Python process.
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
    python -u "$CODE_DIR/test.py"

exit_code=$?


# ============================================================
# END
# ============================================================

end_time=$(date +%s)

echo ""
echo "=========================================="
echo "ANALYSIS FINISHED"
echo "=========================================="
echo "Job ended at: $(date)"

elapsed=$((end_time - start_time))

echo "Elapsed time: $((elapsed / 3600))h $(((elapsed % 3600) / 60))m $((elapsed % 60))s"
echo "Exit code: $exit_code"

exit $exit_code
