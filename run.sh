source env.sh
make
sleep 10
sbatch job_submit_d2q9-bgk
sleep 30
make check
sleep 30
cat d2q9-bgk.out
