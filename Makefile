INSTALLDIR=$(HOME)/common
all: lulesh cg jacobi test-energy-per-rank
cg: c_randdp.c cg.cpp  timers.c mammut_functions.hpp mammut_functions.cpp mammut_config.h 
	mpic++ -g --std=c++11 -I$(INSTALLDIR)/include -o cg  mammut_functions.cpp timers.c c_randdp.c cg.cpp -lm -L$(INSTALLDIR)/lib -lmammut
lulesh: lulesh.cpp timers.c mammut_functions.hpp mammut_functions.cpp mammut_config.h 
	mpic++ -g --std=c++11 -I$(INSTALLDIR)/include -o lulesh mammut_functions.cpp timers.c lulesh.cpp -L$(INSTALLDIR)/lib -lmammut
jacobi: jacobi.cpp header_jacobi.h mammut_functions.hpp mammut_functions.cpp mammut_config.h
	mpic++ -g --std=c++11 -I$(INSTALLDIR)/include -o jacobi mammut_functions.cpp jacobi.cpp -L$(INSTALLDIR)/lib -lmammut
test-energy-per-rank: test-energy-per-rank.cpp mammut_functions.hpp mammut_functions.cpp mammut_config.h
	mpic++ --std=c++11 -I$(INSTALLDIR)/include -o test-energy-per-rank mammut_functions.cpp test-energy-per-rank.cpp -L$(INSTALLDIR)/lib -lmammut
clean:
	rm cg lulesh jacobi
