


#qsub -m beas -M "A.M.Leeuwenberg-15@umcutrecht.nl" -l h_vmem=1G -l h_rt=12:00:00 -pe threaded 1 -e ~/exp_scripts/errors/ -o ~/exp_scripts/out/ jobs_sim.sh



sbatch --cpus-per-task=1 --mail-user="A.M.Leeuwenberg-15@umcutrecht.nl" --error="./errors/slurm-%A_%a.out" --output="./out/slurm-%A_%a.out" --mail-type=END,FAIL --time=4:00:00 --array=0-99:1  1_job_sim.sh

# 1 number of pe threats same as number of bootstraps ?

# 2 memory OK?

# 3 time sufficient?
