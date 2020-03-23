/**
 * Copyright (c) 2016-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * AUTHOR: George Bosilca
 */ 
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "header_jacobi.h"
#include <unistd.h>
#include <vector>
#include <mammut/mammut.hpp>

char** gargv = NULL;

static int NB = -1;
static int MB = -1;
static int P = -1; 
static int Q = -1;
static int KILL_ITER = -1;
int *peer_iters;
mammut::topology::VirtualCore * virtualCore;
mammut::topology::Topology * topology;
mammut::energy::Counter* counter;
mammut::task::ProcessHandler * process;
mammut::cpufreq::CpuFreq* cpufreq;

int generate_border(TYPE* border, int nb_elems)
{
    int i;
    for (i = 0; i < nb_elems; i++) {
        border[i] = (TYPE)(((double) rand()) / ((double) RAND_MAX) - 0.5);
    }
    return 0;
}


int init_matrix(TYPE* matrix, const TYPE* border, int nb, int mb)
{
    int i, j, idx = 0;

    for (idx = 0; idx < nb+2; idx++)
        matrix[idx] = border[idx];
    matrix += idx;

    for (j = 0; j < mb; j++) {
        matrix[0] = border[idx]; idx++;
        for (i = 0; i < nb; i++)
            matrix[1+i] = 0.0;
        matrix[nb+1] = border[idx]; idx++;
        matrix += (nb + 2);
    }

    for (i = 0; i < nb+2; i++)
        matrix[i] = border[idx + i];
    return 0;
}

void parse_arguments(int argc, char **argv) {
    int c;
    while ((c = getopt (argc, argv, "p:q:N:M:f:")) != -1)
        switch (c)
        {
            case 'p':
                P = atoi(optarg);
                break;
            case 'q':
                Q = atoi(optarg);
                break;
            case 'N':
                NB = atoi(optarg);
                break;
            case 'M':
                MB = atoi(optarg);
                break;
            case 'f':
                KILL_ITER = atoi(optarg);
                break;
            case '?':
                if (optopt == 'p' || optopt == 'q' || optopt == 'N' || optopt == 'M' || optopt == 'f') {
                    fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                }
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                            "Unknown option character `\\x%x'.\n",
                            optopt);
                return;
            default:
                abort ();
        }
    if( MB == -1 ) {
        MB = NB;
    }

    if (P == -1 || Q == -1 || NB == -1 || MB == -1) {
        fprintf (stderr, "All options P|Q|NB|MB must be set\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

}

int main( int argc, char* argv[] )
{
    int i, rc, size, rank;
    TYPE *om, *som, *border, epsilon=1e-6;
    MPI_Comm parent;

    gargv = argv;

    mammut::Mammut m;
    topology = m.getInstanceTopology();
    mammut::energy::Energy* energy = m.getInstanceEnergy();
    mammut::task::TasksManager * pm = m.getInstanceTask();
    cpufreq = m.getInstanceCpuFreq();
    process = pm->getProcessHandler(getpid());
if (!NO_MAMMUT) {
    mammut::topology::VirtualCoreId vcid;
    process->getVirtualCoreId(vcid);
    virtualCore = topology->getVirtualCore(vcid);
    counter = energy->getCounter();
    if (counter == NULL) {
        fprintf(stderr, "Mammut does not seem to initialise okay\n");
        exit(-1);
    }
}


    MPI_Init(&argc, &argv);

    parse_arguments(argc, argv);

    MPI_Comm_get_parent( &parent );
    if( MPI_COMM_NULL == parent ) {
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    //printf("Rank %d --> p = %d, q = %d, NB = %d, MB = %d, KILL_ITER = %d\n", rank, P, Q, NB, MB, KILL_ITER);
    /**
     * Ugly hack to allow us to attach with a ssh-based debugger to the application.
     */
    int do_sleep = 0;
    while(do_sleep) {
        sleep(1);
    }

    peer_iters = (int *) malloc(size * sizeof(int));
    for (int i=0; i<size;i++) peer_iters[i] = 0;

    /* make sure we have some randomness */
    border = (TYPE*)malloc(sizeof(TYPE) * 2 * (NB + 2 + MB));
    om = (TYPE*)malloc(sizeof(TYPE) * (NB+2) * (MB+2));
    if( MPI_COMM_NULL == parent ) {
        int seed = rank*NB*MB; srand(seed);
        generate_border(border, 2 * (NB + 2 + MB));
        init_matrix(om, border, NB, MB);
    }
    MPI_Comm_set_errhandler(MPI_COMM_WORLD,
                            MPI_ERRORS_RETURN);

    rc = jacobi_cpu( om, NB, MB, P, Q, MPI_COMM_WORLD, 0 /* no epsilon */, KILL_ITER);

    if( rc < 0 ) {
        printf("The CPU Jacobi failed\n");
        goto cleanup_and_be_gone;
    }

 cleanup_and_be_gone:
    /* free the resources and shutdown */
    free(om);
    free(border);
    free(peer_iters);

    MPI_Finalize();
    return 0;
}
