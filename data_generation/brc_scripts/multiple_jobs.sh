export SCRATCH="/global/scratch/users/riadoshi"
export DATASET="fractal20220817_data"
for (( i=15; i < 16; i++ )); do
    sbatch ${SCRATCH}/vla/brc_scripts/one_job_actions.sh ${i} ${DATASET}
done