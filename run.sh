#!/bin/bash

cd /home/kdichev/message-logging-kernels
    #warmup buillshit
    mpirun --bind-to core /home/kdichev/mammut/build/demo/cpufreq/frequency
    time mpirun --bind-to core $MYPATH/jacobi.global -p 6 -q 4 -N 7500
MYPATH=/home/kdichev/message-logging-kernels
for i in fs #ms std global both
do
    mpirun --bind-to core /home/kdichev/mammut/build/demo/cpufreq/frequency
    mpirun --npernode 2 -n 8 --bind-to core $MYPATH/lulesh.$i -i100
    mv $MYPATH/stats.csv $MYPATH/data/lulesh/stats-$i.csv
    mpirun --bind-to core /home/kdichev/mammut/build/demo/cpufreq/frequency
    mpirun --npernode 4 -n 16 --bind-to core "$MYPATH/cg.$i"
    mv "$MYPATH/stats.csv" "$MYPATH/data/cg/stats-$i.csv"
    mpirun --bind-to core /home/kdichev/mammut/build/demo/cpufreq/frequency
    mpirun --npernode 6 -n 24 --bind-to core $MYPATH/jacobi.$i -p 6 -q 4 -N 7500 -f18
    mv $MYPATH/stats.csv $MYPATH/data/jacobi/stats-$i.csv
done
