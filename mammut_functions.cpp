#include <unistd.h>
#include "mammut_functions.hpp"
#include "mpi.h"

mammut::cpufreq::CpuFreq* Config::cpufreq;
mammut::task::TasksManager *Config::pm;
mammut::topology::VirtualCore * Config::virtualCore;
mammut::topology::Topology* Config::topology;
mammut::energy::Counter* Config::counter;
mammut::energy::CounterCpus* Config::counterCpus;
mammut::energy::Energy*  Config::energy;
mammut::task::ProcessHandler * Config::process;
mammut::Mammut Config::m;

void init_mammut() {

    printf("Initialize Mammut ..\n");
    Config::topology = Config::m.getInstanceTopology();
    Config::energy = Config::m.getInstanceEnergy();
    Config::pm = Config::m.getInstanceTask();
    Config::cpufreq = Config::m.getInstanceCpuFreq();
    Config::process = Config::pm->getProcessHandler(getpid());
    if (!NO_MAMMUT) {
        mammut::topology::VirtualCoreId vcid;
        Config::process->getVirtualCoreId(vcid);
        Config::virtualCore = Config::topology->getVirtualCore(vcid);
        Config::counter = Config::energy->getCounter();
        Config::counterCpus = (mammut::energy::CounterCpus *) Config::energy->getCounter(mammut::energy::COUNTER_CPUS);
        if (Config::counter == NULL) {
            fprintf(stderr, "Mammut does not seem to initialise okay\n");
            exit(-1);
        }
#ifdef SCALE_FREQ_DURING_REC_PSTATE_
        set_12core_max_freq(12, 3199999);
#endif // SCALE_FREQ_DURING_REC_PSTATE_

#ifdef SCALE_MOD_DURING_REC_
        setClockModulation(100.);
#endif // SCALE_MOD_DURING_REC_
        // ALWAYS SET THIS TO MAX FREQUENCY AT THE START !!!
//#ifdef SCALE_FREQ_DURING_REC_
        set_frequency(2400000);
//#endif // SCALE_FREQ_DURING_REC_
    }

}

void set_min_freq_mammut(int cpu) {
    auto domains = Config::cpufreq->getDomains();
    auto dom = domains[cpu];
    dom->removeTurboFrequencies();
    mammut::cpufreq::Frequency target = dom->getAvailableFrequencies().front(); // Use back() for maximum frequency
    dom->setGovernor(mammut::cpufreq::GOVERNOR_USERSPACE);
    dom->setFrequencyUserspace(target);
    //printf("DEBUG: proc %d calls min_freq on cpu %d min. Target = %d at %lf\n", rank, cpu, target, MPI_Wtime()-start);
}

void set_max_freq_mammut(int cpu) {
    //printf("DEBUG: proc %d calls max_freq min\n", rank);
    auto domains = Config::cpufreq->getDomains();
    auto dom = domains[cpu];
    dom->removeTurboFrequencies();
    mammut::cpufreq::Frequency target = dom->getAvailableFrequencies().back(); // Use back() for maximum frequency
    dom->setGovernor(mammut::cpufreq::GOVERNOR_USERSPACE);
    dom->setFrequencyUserspace(target);
    //printf("DEBUG: proc %d calls max_freq on cpu %d min. Target = %d at %lf\n", rank, cpu, target, MPI_Wtime()-start);
}

int get_socket(int rank, int size) {
    return 0; // KOS nodes only !!!!
//    if (rank < size/2) 
//        return 0;
//    else 
//        return 1;
}



void set_frequency(double frequency){
    //Topology* t_handler = m.getInstanceTopology();
    //VirtualCore* vc = t_handler->getVirtualCore(static_cast<VirtualCoreId>(core_id));
    mammut::cpufreq::Domain* d = Config::cpufreq->getDomain(Config::virtualCore);
    d->setGovernor(mammut::cpufreq::GOVERNOR_USERSPACE);
    int iFreq = d->setFrequencyUserspace(frequency);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char hostname[64];
    int resultlen;
    MPI_Get_processor_name(hostname, &resultlen);

    printf("Setting f %lf on host %s, core %u, rank %d ->%d\n", frequency, hostname, Config::virtualCore->getVirtualCoreId(), rank, iFreq);
}

void setClockModulation(double perc) {

    if (!Config::virtualCore->hasClockModulation()) {
        fprintf(stderr, "Clock modulation disabled\n");
        //NO_MAMMUT = true;
    }
    //printf("DEBUG: proc %d calls modulation at %lf in iteration %d with %lf\n", rank, MPI_Wtime(), iteration, perc);
    //
    if (!NO_MAMMUT) {
        // THIS CODE IF YOU WANT TO CHANGE ALL SOCKET CORES!!!
        /*
        for (auto core : topology->getVirtualCores()) {
            core->setClockModulation(perc);
        }
        */
        Config::virtualCore->setClockModulation(perc);
    }


}
void setClockModulationFrac(double fraction) {
    auto values = Config::topology->getCpus().front()->getClockModulationValues();
    double clkModValue;  // This will be set to the actual clock modulation value
    if(values.size()){
        uint32_t index = round((double)(values.size() - 1) * fraction);
        clkModValue = values.at(index);
        setClockModulation(clkModValue);
    }
}


void set_12core_max_freq(int cores, int max_freq) {
    int i;
    for (i = 0; i< cores; i++) {
        FILE *f;  
        char file[64];
        sprintf(file, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", i);
        f = fopen (file, "w");  
        if (f == NULL) {
            fprintf(stderr, "Can't modify CPU properties\n");
            exit(-1);
        }
        //printf("DEBUG: proc %d calls max_freq of %d on all cores at %lf\n", rank, max_freq, MPI_Wtime()-start);
        fprintf (f, "%d", max_freq);
        fclose(f);
    }
}

void down_up(MPI_Comm world, int iteration, int rank, int size) {
    //MPI_Barrier(world);
    if (!NO_MAMMUT) {
#ifdef SCALE_FREQ_DURING_REC_
        set_frequency(1200000);
        // this is needed, because ranks shouldn't
        // end up changing up and down the socket frequency
        //MPI_Barrier(world);
#endif // SCALE_FREQ_DURING_REC_
#ifdef SCALE_MOD_DURING_REC_
        fraction = 0.0625;
        setClockModulationFrac(fraction);
#endif
#ifdef SCALE_FREQ_DURING_REC_PSTATE_
        set_12core_max_freq(12, 1200000);
#endif // SCALE_FREQ_DURING_REC_PSTATE_
        // go back to normal operation
        printf("Rank %d Before barrier in it %d at %lf\n", rank, iteration, MPI_Wtime());

        /*
           int right = (rank+1) % size;
           int left = (size+rank-1) % size;
           int dummy1 = 99;
           int dummy2;
           MPI_Sendrecv(&dummy1, 1, MPI_INT, right, 110, &dummy2, 1, MPI_INT, left, 110, world, MPI_STATUS_IGNORE);
           */
        // This is very important, if this is lower than other's iterations, this
        // will deadlock !!!!

        //int dummy;
        //MPI_Recv(&dummy, 1, MPI_INT, last_dead, 0, world, MPI_STATUS_IGNORE);

        MPI_Barrier(world);
        printf("Rank %d After barrier in it %d at %lf\n", rank, iteration, MPI_Wtime());
#ifdef SCALE_FREQ_DURING_REC_
        set_frequency(2400000);
#endif // SCALE_FREQ_DURING_REC_
#ifdef SCALE_MOD_DURING_REC_
        fraction = 1.0;
        setClockModulationFrac(fraction);
        //printf("Rank %d: Reset clock modulation in iter %d at %lf\n", rank, iteration, MPI_Wtime()-start);
#endif
#ifdef SCALE_FREQ_DURING_REC_PSTATE_
        set_12core_max_freq(12, 3199999);
#endif // SCALE_FREQ_DURING_REC_PSTATE_
        //#endif
    }
}

void log_stats(double *log_joules, double * log_times, int iter, int kill_iter, int size, int failed_proc, MPI_Comm world, int me) {
    // POST_PROCESSING
    
    double *all_joules = NULL;
    double *all_times = NULL;
    if (me == 0) {
        all_joules = (double *) malloc(sizeof(double)*size*iter);
        all_times = (double *) malloc(sizeof(double)*size*iter);
    }
    printf("Rank %d: will call gather with log_joules=%p, iter=%d, all_joules=%p\n", me, log_joules, iter, all_joules);
    MPI_Barrier(world);
    MPI_Gather(log_joules, iter, MPI_DOUBLE, all_joules, iter, MPI_DOUBLE, 0, world);
    MPI_Gather(log_times, iter, MPI_DOUBLE, all_times, iter, MPI_DOUBLE, 0, world);

    if (me == 0) {
        // make sure you add
        // an approximation of what the failed process wasted
        // before the failure (by averaging others)
        // do this both for global and local rollback
        for (int i=0; i<kill_iter; i++) {
            double tmp_sum = 0.;
            double tmp_joul = 0.;
            for (int j=0; j<size; j++) {
                if (j != failed_proc) {
                    tmp_sum += all_times[iter*j+i];
                    tmp_joul += all_joules[iter*j+i];
                }
            }
            int global = (LOG_BFR_DEPTH == 0);
            if (!global) {
                all_times[failed_proc*iter+i] += tmp_sum/(size-1);
                all_joules[failed_proc*iter+i] += tmp_joul/(size-1);
            }
            else {
                all_times[failed_proc*iter+i] += tmp_sum/(2.*(size-1));
                all_joules[failed_proc*iter+i] += tmp_joul/(2.*(size-1));
            }
        }

        std::ofstream f;
        f.open("stats.csv");
        f << "Iteration,Rank,Duration,Joules\n";
        f << "# " << iter << "," << size << "," << kill_iter << "\n";
        for (int i=0; i<iter; i++) {
            for (int j=0; j<size; j++) {
                if (all_times[iter*j+i] == 0.) {printf("Fuck, 0. at %d-%d\n", i,j);}
                f <<  i << "," << j << "," << all_times[iter*j+i] << "," << all_joules[iter*j+i] << std::endl;
            }
        }
        f.close();
    }
}
