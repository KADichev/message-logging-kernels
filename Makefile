all: lulesh cg jacobi_local
cg: c_randdp.c cg.cpp  timers.c mammut_functions.hpp mammut_config.h
	mpic++ --std=c++11 -I$(HOME)/include -o cg  mammut_functions.cpp timers.c c_randdp.c cg.cpp -lm -L$(HOME)/lib -lmammut
lulesh: lulesh.cc timers.c
	mpic++ --std=c++11 -I$(HOME)/include -o lulesh timers.c lulesh.cc -L$(HOME)/lib -lmammut
jacobi_local: jacobi_local.c main_jacobi.c header_jacobi.h mammut_functions.hpp mammut_config.h
	mpic++ --std=c++11 -I$(HOME)/include -o jacobi_local mammut_functions.cpp jacobi_local.c main_jacobi.c -L$(HOME)/lib -lmammut
clean:
	rm cg lulesh jacobi_local
