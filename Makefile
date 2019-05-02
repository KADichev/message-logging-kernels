all: lulesh cg
cg: c_randdp.c cg.c  timers.c
	mpicc -g --std=c99 -o cg timers.c c_randdp.c cg.c -lm
lulesh: lulesh.cc timers.c
	mpic++ -g -o lulesh timers.c lulesh.cc
clean:
	rm cg lulesh
