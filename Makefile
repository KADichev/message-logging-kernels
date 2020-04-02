all: lulesh cg jacobi test-energy-per-rank
cg: c_randdp.c cg.cpp  timers.c mammut_functions.hpp mammut_functions.cpp mammut_config.h 
	mpic++ --std=c++11 -I$(HOME)/include -o cg  mammut_functions.cpp timers.c c_randdp.c cg.cpp -lm -L$(HOME)/lib -lmammut
lulesh: lulesh.cpp timers.c mammut_functions.hpp mammut_functions.cpp mammut_config.h 
	mpic++ -g --std=c++11 -I$(HOME)/include -o lulesh mammut_functions.cpp timers.c lulesh.cpp -L$(HOME)/lib -lmammut
jacobi: jacobi.cpp header_jacobi.h mammut_functions.hpp mammut_functions.cpp mammut_config.h
	mpic++ -g --std=c++11 -I$(HOME)/include -o jacobi mammut_functions.cpp jacobi.cpp -L$(HOME)/lib -lmammut
test-energy-per-rank: test-energy-per-rank.cpp mammut_functions.hpp mammut_functions.cpp mammut_config.h
	mpic++ --std=c++11 -I$(HOME)/include -o test-energy-per-rank mammut_functions.cpp test-energy-per-rank.cpp -L$(HOME)/lib -lmammut
clean:
	rm cg lulesh jacobi
