#include <unistd.h>
#include <limits.h>

#include "mpi.h"
#include "mammut_functions.hpp"
#define DIM 128
#define PINGPONGS 10000000


double test_ping_pong(MPI_Comm world, int rank, int size) {
    int sendto = (rank + 1) % size;
    int recvfrom = (size + rank - 1) % size;
    long send_ping_pong = 0, recv_ping_pong = 0;
    double ping_begin_time = MPI_Wtime();
    while (send_ping_pong < PINGPONGS) {
        MPI_Sendrecv(&send_ping_pong, 1, MPI_LONG, sendto, 0, &recv_ping_pong, 1, MPI_LONG, recvfrom, 0, world, MPI_STATUS_IGNORE);
        send_ping_pong = recv_ping_pong+2;
    }
    double ping_end_time = MPI_Wtime();
    double ping_pong_time = ping_end_time - ping_begin_time;
    return ping_pong_time;
}

/*
void set_min_freq_mammut(int cpu, int rank) {
    std::cout << "Setting cpu " << cpu << " to min\n";
    auto domains = cpufreq->getDomains();
    auto dom1 = domains[rank];
    auto dom2 = domains[rank+24];
    dom1->removeTurboFrequencies();
    dom2->removeTurboFrequencies();
    mammut::cpufreq::Frequency target = dom1->getAvailableFrequencies().front(); // Use back() for maximum frequency
    dom1->setGovernor(mammut::cpufreq::GOVERNOR_USERSPACE);
    dom1->setFrequencyUserspace(target);
    target = dom2->getAvailableFrequencies().front();
    dom2->setGovernor(mammut::cpufreq::GOVERNOR_USERSPACE);
    dom2->setFrequencyUserspace(target);
}

void set_max_freq_mammut(int cpu, int rank) {
    //printf("DEBUG: proc %d calls max_freq min\n", rank);
    auto domains = cpufreq->getDomains();
    auto dom1 = domains[rank];
    auto dom2 = domains[rank+24];
    dom1->removeTurboFrequencies();
    dom2->removeTurboFrequencies();
    mammut::cpufreq::Frequency target = dom1->getAvailableFrequencies().back(); // Use back() for maximum frequency
    dom1->setGovernor(mammut::cpufreq::GOVERNOR_USERSPACE);
    dom1->setFrequencyUserspace(target);
    target = dom2->getAvailableFrequencies().front();
    dom2->setGovernor(mammut::cpufreq::GOVERNOR_USERSPACE);
    dom2->setFrequencyUserspace(target);
}
*/
int main(int argc, char **argv) {


    MPI_Init(&argc, &argv);
    init_mammut();
    int rank, size;
    MPI_Comm world;
    world = MPI_COMM_WORLD;
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &size);

    if (rank == 0) {
        std::cout << "Energy techniques:\n";
#ifdef SCALE_MOD_DURING_REC_
        std::cout << "Clock modulation enabled\n";
#endif // SCALE_MOD_DURING_REC_
#ifdef SCALE_FREQ_DURING_REC_
        std::cout << "Frequency scaling enabled\n";
#endif
#ifdef SCALE_FREQ_DURING_REC_PSTATE_
        std::cout << "Frequency scaling with Interl Pstate enabled\n";
#endif
    }

    double recovery_start_time = MPI_Wtime();
    double recovery_end_time = recovery_start_time;
    Config::counter->reset();
    double a[DIM][DIM];
    double b[DIM][DIM];
    double c[DIM][DIM];
    // PHASE ACTIVE-ACTIVE
   // for (long i=0; i<LONG_MAX;i++) {
   //        auto x = i*i;
   // }
    while ((recovery_end_time-recovery_start_time) < 5) {
        for (int i = 0; i < DIM; i++) {
            for (int j = 0; j < DIM; j++) {
                for (int k = 0; k < DIM; k++) {
                    c[i][j] += a[i][k]*b[k][j];
                }
            }
        }
        recovery_end_time = MPI_Wtime();
    }
    mammut::energy::Joules joules = Config::counter->getJoules();
    if (rank == 0) {
        printf("Rank %d (BURN) : JOULES %lf -- time %lf --average %lf\n", rank, joules, (recovery_end_time-recovery_start_time), joules/(recovery_end_time-recovery_start_time));
    }

    
    // PHASE: <ACTIVE-PASSIVE>
    MPI_Barrier(world);
    recovery_start_time = MPI_Wtime();
    Config::counter->reset();

#ifdef SCALE_MOD_DURING_REC_
        setClockModulation(6.25);
#endif // SCALE_MOD_DURING_REC_
#ifdef SCALE_FREQ_DURING_REC_
        set_frequency(1200000);
#endif // SCALE_FREQ_DURING_REC_
#ifdef SCALE_FREQ_DURING_REC_PSTATE_
    set_12core_max_freq(12, 1200000);
#endif 
    //if (rank == 0) sleep(5);

    if (rank == 0) {
        sleep(5);
    }

    MPI_Barrier(world);
    recovery_end_time = MPI_Wtime();
    joules = Config::counter->getJoules();
    double reduced_ping = test_ping_pong(world, rank, size);
#ifdef SCALE_MOD_DURING_REC_
    setClockModulation(100.);
#endif // SCALE_MOD_DURING_REC_
#ifdef SCALE_FREQ_DURING_REC_
    set_frequency(2400000);
#endif // SCALE_FREQ_DURING_REC_
#ifdef SCALE_FREQ_DURING_REC_PSTATE_
    set_12core_max_freq(12, 3199999);
#endif 

    //if (rank == 0) {
        printf("Rank %d (RECOVERY) : JOULES %lf -- RECOVERY TIME %lf --average %lf\n", rank, joules, (recovery_end_time-recovery_start_time), joules/(recovery_end_time-recovery_start_time)); 
        printf("Rank %d (PING-PONGS count %d) : time = %lf\n", rank, PINGPONGS, reduced_ping);
    //}
    MPI_Finalize();
    return 0;
}
