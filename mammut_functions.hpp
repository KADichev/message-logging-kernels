#include <mammut/mammut.hpp>
#include "mammut_config.h"
#include "mpi.h"

static mammut::cpufreq::CpuFreq* cpufreq;
static mammut::topology::VirtualCore * virtualCore;
static mammut::topology::Topology* topology;
static mammut::energy::Counter* counter;

void set_min_freq_mammut(int cpu);
void set_max_freq_mammut(int cpu);
int get_socket(int rank, int size);
void setClockModulation(double perc);
void setClockModulationFrac(double fraction);
void set_12core_max_freq(int cores, int max_freq);
void down_up(MPI_Comm world, int iteration, int rank, int size);
