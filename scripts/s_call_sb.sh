#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=24:00:00
#SBATCH -J mainJobSubmitter
#SBATCH -p short

# change exp names to experiment needs, each run a python train/test codes
#change paths to your desired locations:
SOURCEPATH=scripts  # call this from a sub folder of main
#CURRDIR=/scratch/username/...
N=2  # pay attention to check recursive numbers
declare -a arr_exp=(
#"D0-SA-1000_nyl"
#"D001-C-1000_nyl"
#"D001-SA-1000_nyl"
#"D001-SA-1000_yyl"
#"D0001-SA-1000_yyl"
#"SR_D0-SA-1000_nyl"
#"SR_D001-C-1000_nyl"
#"SR_D001-SA-1000_nyl"
#"SR_D001-SA-1000_yyl"
"h36m_2d"
#"D001-SA-0100_nyl"
#"D001-SA-0010_nyl"
#"D001-SA-0001_nyl"
                )

for i in "${!arr_exp[@]}"; do
    for j in $(seq 1 $N)
    do
    ##define some job name
#    jobname=$i.job
    ##replace pattern "JOBNAME" in template script to the defined $jobname variable and create a new submit script $CURRDIR/sub.$i.bash:
#    sed "s/JOBNAME/$jobname/g" $SOURCEPATH/sub.template.bash > $CURRDIR/sub.$i.bash
    ##if this is the first job to be submitted, submit without dependancies:

    if [ "$j" -eq "1" ]; then
    ##retrieve the job id number after submitting the created job script:
#    JOBID=`sbatch --job-name=$i-$j ${SOURCEPATH}/${arr_exp[$i]}.sh | sed 's/>//g;s/<//g' | awk '{print $4}'`  # with $i-$j job name
    JOBID=`sbatch ${SOURCEPATH}/${arr_exp[$i]}.sbatch ${1:-46} ${2:-50} | sed 's/>//g;s/<//g' | awk '{print $4}'`  # we can easily see it by same name + jobID
    else
    ## if not the first job, submit this job as a dependent of the previous submitted job:
#    JOBID=`sbatch --job-name=$i-$j --dependency=afterok:${JOBID} ${SOURCEPATH}/${arr_exp[$i]}.sh | sed 's/>//g;s/<//g' | awk '{print $4}'`
    JOBID=`sbatch --dependency=afterok:${JOBID} ${SOURCEPATH}/${arr_exp[$i]}.sbatch ${1:-46} ${2:-50} | sed 's/>//g;s/<//g' | awk '{print $4}'`
    fi
    echo main sbatch job $i ${SOURCEPATH}/${arr_exp[$i]}.sbatch sub $j with ID $JOBID at `date`
    ##sleep for 1 second to let scheduler update job status properly before submitting more jobs:
    sleep 1
    done
done