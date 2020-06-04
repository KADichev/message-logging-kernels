#!/bin/bash

PARAMS="--bind-to core --hostfile $HOME/hostfile --prefix $HOME/common"
MYPATH=/home/kdichev/message-logging-kernels
cd /home/kdichev/message-logging-kernels
    #warmup buillshit
    mpirun $PARAMS /home/kdichev/mammut/build/demo/cpufreq/frequency
    #time mpirun $PARAMS $MYPATH/jacobi.global -p 18 -q 8 -N 1000
for i in global std
do
    mpirun $PARAMS /home/kdichev/mammut/build/demo/cpufreq/frequency
    mpirun $PARAMS --npernode 8 -n 8  $MYPATH/lulesh.$i -i100
    mv $MYPATH/stats.csv $MYPATH/data/noft/lulesh/8/stats-$i.csv
    mpirun $PARAMS --npernode 8 -n 125  $MYPATH/noft/lulesh.$i -i100
    mv $MYPATH/stats.csv $MYPATH/data/noft/lulesh/125/stats-$i.csv
    mpirun $PARAMS /home/kdichev/mammut/build/demo/cpufreq/frequency
    mpirun $PARAMS --npernode 8 -n 16 $MYPATH/cg.$i
    mv $MYPATH/stats.csv $MYPATH/data/noft/cg/16/stats-$i.csv
    mpirun $PARAMS --npernode 8 -n 64 $MYPATH/cg.$i
    mv $MYPATH/stats.csv $MYPATH/data/noft/cg/64/stats-$i.csv
    mpirun $PARAMS /home/kdichev/mammut/build/demo/cpufreq/frequency
    mpirun $PARAMS --npernode 8 -n 16  $MYPATH/jacobi.$i -p 4 -q 4 -N 7500
    mv $MYPATH/stats.csv $MYPATH/data/noft/jacobi/16/stats-$i.csv
   # mpirun $PARAMS --npernode 8 -n 144 $MYPATH/jacobi.$i -p 12 -q 12 -N 7500 -f18
   # mv $MYPATH/stats.csv $MYPATH/data/noft/jacobi/144/stats-$i.csv
done
