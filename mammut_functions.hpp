#include <mammut/mammut.hpp>
#include "mammut_config.h"
#include "mpi.h"

class Config {
    public:
        static mammut::cpufreq::CpuFreq* cpufreq;
        static mammut::task::TasksManager *pm;
        static mammut::topology::VirtualCore * virtualCore;
        static mammut::topology::Topology* topology;
        static mammut::energy::Counter* counter;
        static mammut::energy::Energy*  energy;
        static mammut::task::ProcessHandler * process;
        static mammut::Mammut m;
};
static double fraction = 1.0;

void init_mammut();
void set_frequency(double frequency);
void set_min_freq_mammut(int cpu);
void set_max_freq_mammut(int cpu);
int get_socket(int rank, int size);
void setClockModulation(double perc);
void setClockModulationFrac(double fraction);
void set_12core_max_freq(int cores, int max_freq);
void down_up(MPI_Comm world, int iteration, int rank, int size);
void log_stats(double *log_joules, double * log_times, int iter, int kill_iter, int size, int failed_proc, MPI_Comm world, int me);
