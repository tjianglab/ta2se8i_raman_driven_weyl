#!/bin/bash

#find out how many items as n
n=`find -name "disp-*" | wc -l`
echo "$n"
#create new file
> Status_Jobs

for mode in {001..034}
#for mode in `seq -w 1 "$n"` #key words
do
cd disp-$mode

echo 'Disp '"$mode"
FILE=OUTCAR
if test -f "$FILE"; then
        echo "$FILE exist"

        if grep "General timing and accounting informations for this job" "$FILE"; then
                echo "JOB DONE"
                #rm Status_Jobs
                paste <(printf %s "$mode") <(printf %s 'JOB DONE') >> ../Status_Jobs
        else
                echo "JOB NOT DONE!!!!!!"
                #rm OUTCAR
                #rm Status_Jobs
                paste <(printf %s "$mode") <(printf %s 'JOB NOT DONE') >> ../Status_Jobs
                #cat ../job | sed 's/XX1/'$mode'/' > job
                #sbatch --partition=preemptable --qos=preemptable job
                #sbatch job
                #sbatch -q scavenger --time-min=0:05:00 job
        fi
else
        echo "$FILE not found!!!"
        #rm Status_Jobs
        paste <(printf %s "$mode") <(printf %s 'FILE NOT FOUND') >> ../Status_Jobs
        #cat ../job | sed 's/XX1/'$mode'/' > job
        #sbatch --partition=preemptable --qos=preemptable job
        #sbatch job
        #sbatch -q scavenger --time-min=0:05:00 job
fi
cd ../
done

