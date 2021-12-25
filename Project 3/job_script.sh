#!/bin/bash
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#

#SBATCH --job-name=bfs-jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --output=bfs-job.out

################################################################################
##################### !!! DO NOT EDIT ABOVE THIS LINE !!! ######################
################################################################################
# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a

echo "Running Job...!"
echo "==============================================================================="
echo "Running compiled binary..."

lscpu

#parallel version
echo "Parallel version with 1 process"
mpirun -np 1 ./bfs graphs/mycielskian14.mtx -1 1

echo "Parallel version with 2 processes"
mpirun -np 2 ./bfs graphs/mycielskian14.mtx -1 1

echo "Parallel version with 4 processes"
mpirun -np 4 ./bfs graphs/mycielskian14.mtx -1 1

echo "Parallel version with 8 processes"
mpirun -np 8 ./bfs graphs/mycielskian14.mtx -1 1

echo "Parallel version with 16 processes"
mpirun -np 16 ./bfs graphs/mycielskian14.mtx -1 1

