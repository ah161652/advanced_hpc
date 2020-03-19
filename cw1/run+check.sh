source env.sh
make
sleep 10
sbatch job_submit_d2q9-bgk
sleep 90
cat d2q9-bgk.out
make check
sleep 60
cat d2q9-bgk.out
