/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2013-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include <unistd.h> //sleep

#include <math.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>
#include "header_jacobi.h"
#include <vector>
#include <mammut/mammut.hpp>
#include "mammut_functions.hpp"

//#define min(a,b) (a<b?a:b)


void set_12core_max_freq(int cores, int max_freq);
void setClockModulationFrac(double fraction);
std::map<std::pair<int, std::pair<int,int> >, void *> logs;
double start_time;
int min_it, max_it = -1;
static double total_wait = 0.;
static double total_sor = 0.;
static double total_joules = 0.;
static double start;
static  int SOR_updates = 0;
static int MPIX_Comm_replace(MPI_Comm comm, MPI_Comm *newcomm);

static TYPE *bckpt = NULL;
static TYPE *lckpt = NULL;
static bool post_failure = false;
static bool post_failure_sync = false;
static int rank = MPI_PROC_NULL, verbose = 1; /* makes this global (for printfs) */
static char estr[MPI_MAX_ERROR_STRING]=""; static int strl; /* error messages */

extern char** gargv;
extern int* peer_iters;

static int ckpt_iteration = -1, last_dead = FAILED_RANK;
static int iteration;
//static MPI_Comm ew, ns;

static jmp_buf stack_jmp_buf;

void replay(MPI_Comm comm, double * matrix, int NB, int MB, int P, int Q);
int send_wrapper(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                  MPI_Comm comm, MPI_Request * request, int replay_it, int stage);
bool am_i_involved(int ns_rank, int ew_rank, int rvrs_it, int P, int Q, int NB, int MB);
//int do_recovering_jacobi(int NB, int MB, int P, MPI_Comm comm, int failed_iteration, int failed_rank, TYPE *matrix, TYPE *nm);

#define CKPT_STEP 20


void print_waits(double *total_waits, double *total_sors, int size, int P, int Q) {
    if (rank == 0) {
        printf("[");
        for (int i=0; i<size; i++) {
            if (i % Q == 0) printf("[");
            printf("%lf", total_waits[i]);
            if (i % P == P-1 && i < (size-1))
                printf("],");
            else if (i == size-1) 
                printf("]");
            else 
                printf(",");
            //if (i % Q == Q-1) printf("]");
        }
        printf("]\n\n");

        printf("[");
        for (int i=0; i<size; i++) {
            if (i % Q == 0) printf("[");
            printf("%lf", total_sors[i]);
            if (i % P == P-1 && i < (size-1))
                printf("],");
            else if (i == size-1) 
                printf("]");
            else 
                printf(",");
            //if (i % Q == Q-1) printf("]");
        }
        printf("]\n");

        free(total_waits);
        free(total_sors);
    }
}


/* FORMULA:
   involved(rank, failed, failed_it) :=
   (failed_it % CP_INT ≥ Nx∗(|X(rank)−X(failed)|−1) (2) AND
   failed_it % CP_INT ≥ Ny ∗(|Y(rank)−Y(failed)|−1)
   )
    */
bool am_i_involved(int ns_rank, int ew_rank, int rvrs_it, int P, int Q, int NB, int MB) {

    // translate failed_rank on 2D map
    int failed_ns = FAILED_RANK / P;
    int failed_ew =  FAILED_RANK % P;
    //printf("DEBUG: failed_ns = %d, failed_ew = %d\n", failed_ns, failed_ew);
    //printf("%d - %d <= %d/%d+1 && (%d - %d) <= %d/%d+1",ns_rank, failed_ns, rvrs_it, elems_ns, ew_rank, failed_ew, rvrs_it, elems_ew);
    bool termX = (rvrs_it >= ((abs(ew_rank-failed_ew) - 1) * MB));
    bool termY = (rvrs_it >= ((abs(ns_rank-failed_ns) - 1) * NB));
    return termX && termY;
}





/* world will swap between worldc[0] and worldc[1] after each respawn */
static MPI_Comm worldc[2] = { MPI_COMM_NULL, MPI_COMM_NULL };
static int worldi = 0;

#define world (worldc[worldi])

/* repair comm world, reload checkpoints, etc...
*  Return: true: the app needs to redo some iterations
*          false: no failure was fixed, we do not need to redo any work.
*/
static int app_needs_repair(MPI_Comm comm, int rank)
{
post_failure = true;
printf("Enter app needs repair, say rank %d\n", rank);
/* This is the first time we see an error on this comm, do the swap of the
 * worlds. Next time we will have nothing to do. */
if( comm == world ) {
    /* swap the worlds */
    worldi = (worldi+1)%2;
    /* We keep comm around so that the error handler remains attached until the
     * user has completed all pending ops; it is expected that the user will
     * complete all ops on comm before posting new ops in the new world.
     * Beware that if the user does not complete all ops on comm and the handler
     * is invoked on the new world inbetween, comm may be freed while
     * operations are still pending on it, and a fatal error may be
     * triggered when these ops are finally completed (possibly in Finalize)*/
    if( MPI_COMM_NULL != world ) MPI_Comm_free(&world);
    MPIX_Comm_replace(comm, &world);

    if( MPI_COMM_NULL == comm ) return false; /* ok, we repaired nothing, no need to redo any work */
    _longjmp( stack_jmp_buf, 1 );

    // ToDo: shouldn't have to do that here?
    //if( MPI_COMM_NULL == comm ) return false; /* ok, we repaired nothing, no need to redo any work */
//	int rank;
//	MPI_Comm_rank(world, &rank);
//	printf("Rank %d will longjump\n", rank);
}
return true; /* we have repaired the world, we need to reexecute */
}

/* Do all the magic in the error handler */
static void errhandler_respawn(MPI_Comm* pcomm, int* errcode, ...)
{
int eclass;
MPI_Error_class(*errcode, &eclass);

if( verbose ) {
    MPI_Error_string(*errcode, estr, &strl);
    fprintf(stderr, "%04d: errhandler invoked with error %s\n", rank, estr);
}

if( MPIX_ERR_PROC_FAILED != eclass &&
    MPIX_ERR_REVOKED != eclass ) {
    MPI_Abort(MPI_COMM_WORLD, *errcode);
}
int rank;
MPI_Comm_rank(world, &rank);
printf("Rank %d: WILL CALL REVOKE\n", rank);
//MPIX_Comm_revoke(ew);
//MPIX_Comm_revoke(ns);
MPIX_Comm_revoke(world);

app_needs_repair(world, rank);
}

void print_timings( MPI_Comm scomm,
                double twf )
{
/* Storage for min and max times */
double mtwf, Mtwf;

MPI_Reduce( &twf, &mtwf, 1, MPI_DOUBLE, MPI_MIN, 0, scomm );
MPI_Reduce( &twf, &Mtwf, 1, MPI_DOUBLE, MPI_MAX, 0, scomm );

if( 0 == rank ) printf( "## Timings ########### Min         ### Max         ##\n"
                        "Loop    (w/ fault)  # %13.5e # %13.5e\n",
                        mtwf, Mtwf );
}

static int MPIX_Comm_replace(MPI_Comm comm, MPI_Comm *newcomm)
{
MPI_Comm icomm, /* the intercomm between the spawnees and the old (shrinked) world */
    scomm, /* the local comm for each sides of icomm */
    mcomm; /* the intracomm, merged from icomm */
MPI_Group cgrp, sgrp, dgrp;
int rc, flag, rflag, i, nc, ns, nd, crank, srank, drank;

redo:
if( comm == MPI_COMM_NULL ) { /* am I a new process? */
    /* I am a new spawnee, waiting for my new rank assignment
     * it will be sent by rank 0 in the old world */
    MPI_Comm_get_parent(&icomm);
    scomm = MPI_COMM_WORLD;
    MPI_Recv(&crank, 1, MPI_INT, 0, 1, icomm, MPI_STATUS_IGNORE);
    last_dead = crank;

    if( verbose ) {
        MPI_Comm_rank(scomm, &srank);
        printf("Spawnee %d: crank=%d\n", srank, crank);
    }
} else {
    /* I am a survivor: Spawn the appropriate number
     * of replacement processes (we check that this operation worked
     * before we procees further) */
    /* First: remove dead processes */
    MPIX_Comm_shrink(comm, &scomm);
    MPI_Comm_size(scomm, &ns);
    MPI_Comm_size(comm, &nc);
    nd = nc-ns; /* number of deads */
    if( 0 == nd ) {
        /* Nobody was dead to start with. We are done here */
        MPI_Comm_free(&scomm);
        *newcomm = comm;
        return MPI_SUCCESS;
    }
    /* We handle failures during this function ourselves... */
    MPI_Comm_set_errhandler( scomm, MPI_ERRORS_RETURN );

    printf("Rank %d: about to respawn someone\n", rank);
    rc = MPI_Comm_spawn(gargv[0], &gargv[1], nd, MPI_INFO_NULL,
                        0, scomm, &icomm, MPI_ERRCODES_IGNORE);
    printf("Rank %d: after respawn someone\n", rank);
    flag = (MPI_SUCCESS == rc);
    MPIX_Comm_agree(scomm, &flag);
    if( !flag ) {
        if( MPI_SUCCESS == rc ) {
            MPIX_Comm_revoke(icomm);
            MPI_Comm_free(&icomm);
        }
        MPI_Comm_free(&scomm);
        if( verbose ) fprintf(stderr, "%04d: comm_spawn failed, redo\n", rank);
        goto redo;
    }

    /* remembering the former rank: we will reassign the same
     * ranks in the new world. */
    MPI_Comm_rank(comm, &crank);
    MPI_Comm_rank(scomm, &srank);
    /* the rank 0 in the scomm comm is going to determine the
     * ranks at which the spares need to be inserted. */
    if(0 == srank) {
        /* getting the group of dead processes:
         *   those in comm, but not in scomm are the deads */
        MPI_Comm_group(comm, &cgrp);
        MPI_Comm_group(scomm, &sgrp);
        MPI_Group_difference(cgrp, sgrp, &dgrp);
        /* Computing the rank assignment for the newly inserted spares */
        for(i=0; i<nd; i++) {
            MPI_Group_translate_ranks(dgrp, 1, &i, cgrp, &drank);
            /* sending their new assignment to all new procs */
            MPI_Send(&drank, 1, MPI_INT, i, 1, icomm);
            // left border (1 failed)
            last_dead = drank;
        }
        MPI_Group_free(&cgrp); MPI_Group_free(&sgrp); MPI_Group_free(&dgrp);
    }

}

/* Merge the intercomm, to reconstruct an intracomm (we check
 * that this operation worked before we proceed further) */
rc = MPI_Intercomm_merge(icomm, 1, &mcomm);
rflag = flag = (MPI_SUCCESS==rc);
MPIX_Comm_agree(scomm, &flag);
if( MPI_COMM_WORLD != scomm ) MPI_Comm_free(&scomm);
MPIX_Comm_agree(icomm, &rflag);
MPI_Comm_free(&icomm);
if( !(flag && rflag) ) {
    if( MPI_SUCCESS == rc ) {
        MPI_Comm_free(&mcomm);
    }
    if( verbose ) fprintf(stderr, "%04d: Intercomm_merge failed, redo\n", rank);
    goto redo;
}

/* Now, reorder mcomm according to original rank ordering in comm
 * Split does the magic: removing spare processes and reordering ranks
 * so that all surviving processes remain at their former place */
rc = MPI_Comm_split(mcomm, 1, crank, newcomm);

/* Split or some of the communications above may have failed if
 * new failures have disrupted the process: we need to
 * make sure we succeeded at all ranks, or retry until it works. */
flag = (MPI_SUCCESS==rc);
MPIX_Comm_agree(mcomm, &flag);
MPI_Comm_free(&mcomm);
if( !flag ) {
    if( MPI_SUCCESS == rc ) {
        MPI_Comm_free( newcomm );
    }
    if( verbose ) fprintf(stderr, "%04d: comm_split failed, redo\n", rank);
    goto redo;
}

/* restore the error handler */
if( MPI_COMM_NULL != comm ) {
    MPI_Errhandler errh;
    MPI_Comm_get_errhandler( comm, &errh );
    MPI_Comm_set_errhandler( *newcomm, errh );
}
int loc_rank;
MPI_Comm_rank(*newcomm, &loc_rank);
//printf("Done with the recovery (rank %d)\n", loc_rank);

return MPI_SUCCESS;
}

/**
* We are using a Successive Over Relaxation (SOR)
* http://www.physics.buffalo.edu/phy410-505/2011/topic3/app1/index.html
*/
TYPE SOR1( TYPE* nm, TYPE* om,
       int nb, int mb )
{
TYPE norm = 0.0;
TYPE _W = 2.0 / (1.0 + M_PI / (TYPE)nb);
int i, j, pos;

TYPE dumb = 0.0;
for(j = 0; j < mb; j++) {
    for(i = 0; i < nb; i++) {
        pos = 1 + i + (j+1) * (nb+2);
        dumb += om[pos];
        nm[pos] = (1 - _W) * om[pos] +
                  _W / 4.0 * (nm[pos - 1] +
                              om[pos + 1] +
                              nm[pos - (nb+2)] +
                              om[pos + (nb+2)]);
        norm += (nm[pos] - om[pos]) * (nm[pos] - om[pos]);
    }
}
return norm;
}

int preinit_jacobi_cpu(void)
{
return 0;
}

int jacobi_cpu(TYPE* matrix, int NB, int MB, int P, int Q, MPI_Comm comm, TYPE epsilon, int KILL_ITER)
{

double log_joules[MAX_ITER];
double log_times[MAX_ITER];
start_time = MPI_Wtime();
int dbg_counter;
int  allowed_to_kill = (KILL_ITER != -1);
double sum = 0.;
double start_this_it;
int size, ew_rank, ew_size, ns_rank, ns_size;
TYPE *om, *nm, *tmpm, *send_east, *send_west, *recv_east, *recv_west, diff_norm;
start = 0.;
double twf=0; /* timings */
MPI_Errhandler errh;
MPI_Comm parent;
MPI_Request req[8] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                      MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
double start_exch, end_exch;

MPI_Comm_create_errhandler(&errhandler_respawn, &errh);
/* Am I a spare ? */
MPI_Comm_get_parent( &parent );
om = matrix;
if( MPI_COMM_NULL == parent ) {
    /* First run: Let's create an initial world,
     * a copy of MPI_COMM_WORLD */
    MPI_Comm_dup( comm, &world );
} else {
    allowed_to_kill = 0;
    ckpt_iteration = 0;//MAX_ITER;
    /* I am a spare, lets get the repaired world */
    printf("I am a SPARE\n");
    app_needs_repair(MPI_COMM_NULL, -1);
}

MPI_Comm_rank(world, &rank);
MPI_Comm_size(world, &size);
    printf("Rank %d => NB = %d, MB = %d\n", rank, NB, MB);
//printf("rank %d -- KILL_ITER = %d, allowed_to_kill = %d\n", rank, KILL_ITER, allowed_to_kill);

nm = (TYPE*)malloc(sizeof(TYPE)*(NB+2) * (MB+2));
//nm_tmp = (TYPE*)malloc(sizeof(TYPE)*(NB+2) * (MB+2));
//om_tmp = (TYPE*)malloc(sizeof(TYPE)*(NB+2) * (MB+2));
send_east = (TYPE*)malloc(sizeof(TYPE) * MB);
send_west = (TYPE*)malloc(sizeof(TYPE) * MB);
recv_east = (TYPE*)malloc(sizeof(TYPE) * MB);
recv_west = (TYPE*)malloc(sizeof(TYPE) * MB);

/**
 * Prepare the space for the buddy ckpt.
 */


iteration = 0;

restart:  /* This is my restart point */
bckpt = (TYPE*)malloc(sizeof(TYPE) * (NB+2) * (MB+2));
lckpt = (TYPE*)malloc(sizeof(TYPE) * (NB+2) * (MB+2));
start = MPI_Wtime();
double start_it, end_it;
int do_recover = _setjmp(stack_jmp_buf);

//printf("Rank %d: after setjmp\n", rank);
ew_size = P ;
ns_size = Q ;
ns_rank = rank / P;
ew_rank = rank % P;
/* We set an errhandler on world, so that a failure is not fatal anymore. */
MPI_Comm_set_errhandler( world, errh );

/* create the north-south and east-west communicator */
//MPI_Comm_split(world, rank % P, rank, &ns);
//MPI_Comm_size(ns, &ns_size);
//MPI_Comm_rank(ns, &ns_rank);
//ns_rank 
//MPI_Comm_split(world, rank / P, rank, &ew);
//MPI_Comm_size(ew, &ew_size);
//MPI_Comm_rank(ew, &ew_rank);

int rc;
//set_12core_max_freq(12, 2400000);

for (; iteration<MAX_ITER; iteration++) {
    //printf("Rank %d: start iteration %d, max_iter = %d last_dead = %d\n", rank, iteration, max_it, last_dead);


    if (!NO_MAMMUT) {
        start_it = MPI_Wtime();
        Config::counter->reset();
    }

    // super important to not deadlock
    // since restarted process will signal the rest to continue later
    if (post_failure)  {

        post_failure = false;
        post_failure_sync = true;
        //printf("Rank %d: before replay at %lf\n", rank, MPI_Wtime());
        replay(world, om, NB, MB, P, Q);
        //int color = (rank == last_dead);
        //MPI_Comm newcomm;
        //printf("Rank %d: before split\n", rank);
        //MPI_Comm_split(world, color, 1, &newcomm);
        //printf("Rank %d: after split\n", rank);
        //MPI_Comm_free(&newcomm);
        printf("Rank %d: after replay at iteration %d\n", rank, iteration);


    }
    if (post_failure_sync) {
        if (iteration == max_it) {
            post_failure_sync = false;
            down_up(world, iteration, rank, size);
        }

    }
  //  else if (iteration == max_it) {
  //      printf("Rank %d will call barrier\n", rank);
  //      MPI_Barrier(world);
  //  }
  
    dbg_counter = 0;
    start_this_it = MPI_Wtime();
    //printf("Rank %d: time since start in iter %d: %lf\n", rank, iteration, start_this_it-start);

  //if (post_failure)  {
//    double recovery_start_time = MPI_Wtime();
//    int color;
//    if (!NO_MAMMUT) {
//        //start_it = MPI_Wtime();
//        counter->reset();
//    }
//    //local_rec(om, nm, world, ns_rank, ew_rank, NB, MB, P, Q, KILL_ITER, ns_size, ew_size, color);
//    replay(world, om, NB, MB, P, Q);
//
//    double recovery_end_time = MPI_Wtime();

    //if (!NO_MAMMUT) { mammut::energy::Joules joules = counter->getJoules();
        //printf("Rank %d -- RECOVERY JOULES %lf -- RECOVERY TIME %lf, average -> %lf \n", rank, joules, (recovery_end_time-recovery_start_time), joules/(recovery_end_time-recovery_start_time));

    //}

    // some ranks are modified after local_rec (don't know why?), need to be fixed:
//    MPI_Comm_rank(world, &rank);
//    post_failure = false;
  //}

  
  dbg_counter = 0;
  start_this_it = MPI_Wtime();
  //printf("Rank %d: time since start in iter %d: %lf\n", rank, iteration, start_this_it-start);
    
    int i;int j;
    sum = 0.;

    /**
     * If we are at the right point in time, let's kill a process...
     */
    if( allowed_to_kill && (KILL_ITER == iteration) ) {  /* original execution */
      allowed_to_kill = 0;
      if( FAILED_RANK == rank ) {
        printf("Before crash value: %lf %lf %lf\n", om[0], om[1], om[2]);
        raise(SIGKILL);
      }
    }


    if( 0 != ns_rank ) {
      MPI_Irecv( RECV_NORTH(om), NB, MPI_TYPE, rank - P, 0, world, &req[0]);
      send_wrapper( SEND_NORTH(om), NB, MPI_TYPE, rank - P, 0, world, &req[1], iteration, 0);
    }
    if( (ns_size-1) != ns_rank ) {
      MPI_Irecv( RECV_SOUTH(om), NB, MPI_TYPE, rank + P, 0, world, &req[2]);
      send_wrapper( SEND_SOUTH(om), NB, MPI_TYPE, rank + P, 0, world, &req[3], iteration, 1);
    }

    for(i = 0; i < MB; i++) {
      send_west[i] = om[(i+1)*(NB+2)      + 1];  /* the real local data */
      send_east[i] = om[(i+1)*(NB+2) + NB + 0];  /* not the ghost region */
    }

    if( (ew_size-1) != ew_rank ) {
      MPI_Irecv( recv_east,      MB, MPI_TYPE, rank + 1, 0, world, &req[4]);
      send_wrapper( send_east,      MB, MPI_TYPE, rank + 1, 0, world, &req[5], iteration, 2);
    }
    if( 0 != ew_rank ) {
      MPI_Irecv( recv_west,      MB, MPI_TYPE, rank - 1, 0, world, &req[6]);
      send_wrapper( send_west,      MB, MPI_TYPE, rank - 1, 0, world, &req[7], iteration, 3);
    }
    //MPI_Waitall(8, req, MPI_STATUSES_IGNORE);


//    /* wait until they all complete */
//    dbg_counter = 0;
//    for (int i=0; i<8; i++) {
//        if (req[i] != MPI_REQUEST_NULL) dbg_counter++;
//
//    }
    //printf("Rank %d: neighbour comm (normal) = %d\n", rank, dbg_counter);
    /* wait until they all complete */
    start_exch = MPI_Wtime();


    //printf("Rank %d: iter %d calling waitall\n", rank, iteration);
 //   MPI_Status status;
 //   int cnt;
 //   int flag = 0;
 //   while (cnt < dbg_counter) {
 //       cnt = 0;
 //       for (int i=0; i<8; i++) {
 //           if (req[i] != MPI_REQUEST_NULL) {
 //               MPI_Test(&req[i], &flag, &status);
 //               if (flag) cnt++;
 //           }
 //       }
 //       sleep(0.02);
 //   }
 //
    /* post receives from the neighbors */

    //printf("Rank %d: normal iter %d start waitall\n", rank, iteration);
    rc = MPI_Waitall(8, req, MPI_STATUSES_IGNORE);
    //printf("Rank %d: normal iter %d finishing waitall\n", rank, iteration);
    end_exch = MPI_Wtime();
    total_wait += (end_exch-start_exch);

    //printf("Rank %d done with sends/recvs at iteration %d\n", rank, iteration);
    for(i = 0; i < MB; i++) {
      om[(i+1)*(NB+2)         ] = recv_west[i];
      om[(i+1)*(NB+2) + NB + 1] = recv_east[i];
    }


do_sor:
    /* replicate the east-west newly received data */
    for(i = 0; i < MB; i++) {
        nm[(i+1)*(NB+2)         ] = om[(i+1)*(NB+2)         ];
        nm[(i+1)*(NB+2) + NB + 1] = om[(i+1)*(NB+2) + NB + 1];
    }
    /* replicate the north-south neighbors */
    for(i = 0; i < NB; i++) {
        nm[                    i + 1] = om[                    i + 1];
        nm[(NB + 2)*(MB + 1) + i + 1] = om[(NB + 2)*(MB + 1) + i + 1];
    }


    /**
     * Call the Successive Over Relaxation (SOR) method
     */
    double begin_sor1 = MPI_Wtime();
    diff_norm = SOR1(nm, om, NB, MB);
    total_sor += MPI_Wtime()-begin_sor1;
    //printf("STANDARD ITER: rank %d, diff_norm = %lf\n", rank, diff_norm);
    SOR_updates++;

    //if(verbose)
        //printf("Rank %d norm %f at iteration %d - [0]=%.16lf, [1]=%.16lf, [2]=%.16lf\n", rank, diff_norm, iteration, om[0], om[5], om[10]);
    //rc = MPI_Allreduce(MPI_IN_PLACE, &diff_norm, 1, MPI_TYPE, MPI_SUM,
                  //world);

    //if(0 == rank) {
        //printf("Rank %d: Iteration %4d norm %f time %lf\n", rank, iteration, sqrtf(diff_norm), MPI_Wtime()-start);
    //}
    sum = 0.;
    for(j = 0; j < MB; j++) {
      for(i = 0; i < NB; i++) {
        int pos = 1 + i + (j+1) * (NB+2);
        sum+= nm[pos]*nm[pos];
      }
    }
    //MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, world);

    //if (0 == rank) printf("My norm: %f at end of iteration %d\n", sum, iteration);

    tmpm = om; om = nm; nm = tmpm;  /* swap the 2 matrices */

    if (peer_iters[rank] != iteration) {
        printf("Rank %d: something is seriosly wrong -> peer_iters[rank] = %d = iteration = %d\n", rank, peer_iters[rank], iteration);
        MPI_Abort(world, -1);
    }

    int min_elem = *std::min_element(peer_iters,peer_iters+size);
    for (int i=0; i<size; i++) {
        if (min_elem == peer_iters[i]) peer_iters[i]++;
    }

//    if (!NO_MAMMUT) {
//        end_it = MPI_Wtime();
//        mammut::energy::Joules joules = counter->getJoules();
//        //counter->reset();
//        printf("Rank %d -- Iter %d -- Joules %f - average -> %lf\n", rank, iteration, joules, joules/(end_it-start_it));
//    }

    /**
     * Every XXX iterations do a checkpoint.
     */
    if( (0 == (iteration % CKPT_STEP)) ) {

      // buddy ckpt
        if (size > 1) {
            int buddy = (rank % 2 == 0)?rank+1:rank-1;
            MPI_Sendrecv(om, (NB+2)*(MB+2), MPI_TYPE, buddy, 111, bckpt, (NB+2)*(MB+2), MPI_TYPE, buddy, 111, world, MPI_STATUS_IGNORE);
            memcpy(lckpt, om, sizeof(TYPE)*(NB+2)*(MB+2));
            ckpt_iteration = iteration;
        }
    }

    if (!NO_MAMMUT) {
        end_it = MPI_Wtime();
        mammut::energy::Joules joules = Config::counter->getJoules();
        total_joules += joules;
        log_joules[iteration] = joules;
        log_times[iteration] = (end_it - start_it);
        //printf("it %d rank %d -> %lf\n", iteration, rank, log_times[iteration]);
    }

} 

twf = MPI_Wtime() - start;
print_timings( world, twf );
int total_updates = 0;
//MPI_Reduce(&SOR_updates, &total_updates, 1, MPI_INT, MPI_SUM, 0, world);
//if (rank == 0) {
char hostname[64];
int resultlen;
double * total_waits = NULL;
double * total_sors = NULL;
if (rank == 0) total_waits = (double *) malloc(sizeof(double)*size);
if (rank == 0) total_sors = (double *) malloc(sizeof(double)*size);
MPI_Get_processor_name(hostname, &resultlen);
    printf("Rank %d on %s: Total operations %d, joules = %lf, waiting time = %lf\n", rank, hostname, SOR_updates, total_joules, total_wait);
    MPI_Gather(&total_wait, 1, MPI_DOUBLE, total_waits, 1, MPI_DOUBLE, 0, world);
    MPI_Gather(&total_sor, 1, MPI_DOUBLE, total_sors, 1, MPI_DOUBLE, 0, world);

    //}

    //free(om); <= SEG FAULT
    free(send_west);
    free(send_east);
    free(recv_west);
    free(recv_east);
    free(bckpt);

//    char filename[12];
//    sprintf(filename,"stats.%d.csv",rank);
//    FILE *f2 = fopen(filename,"w");
//    for (int i=0; i<iteration; i++) {
//        fprintf(f2, "%d,%lf,%lf\n", i, log_times[i], log_joules[i]);;
//    }
//fclose(f2);


log_stats(&log_joules[0], &log_times[0], iteration, KILL_ITER, size, FAILED_RANK, world, rank);

    return iteration;
}

void replay(MPI_Comm comm, double * matrix, int NB, int MB, int P, int Q) {

    int buddy = (rank % 2 == 0)?rank+1:(rank-1);
    int size;
    MPI_Comm_size(comm, &size);

    bool snd_bkp = (buddy == last_dead) ;
    bool rcv_bkp = (rank == last_dead) ;
    if (snd_bkp) {
        printf("Sending checkpoint: %d -> %d\n", rank, buddy);
        MPI_Send(bckpt, (NB+2)*(MB+2), MPI_TYPE, buddy, 111, comm);
        MPI_Send(&ckpt_iteration, 1, MPI_INT, buddy, 111, comm);

    }
    if (rcv_bkp)  {
        printf("Receiving checkpoint: %d <- %d\n", rank, buddy);
        MPI_Recv(matrix, (NB+2)*(MB+2), MPI_TYPE, buddy, 111, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&iteration, 1, MPI_INT, buddy, 111, comm, MPI_STATUS_IGNORE);
        iteration++;
    }

    //printf("Rank %d: before allgather sending iteration = %d\n", rank, iteration);
    MPI_Allgather(&iteration, 1, MPI_INT, peer_iters, 1, MPI_INT, comm);
    //printf("Rank %d: after allgather\n", rank);

    min_it = *std::min_element(peer_iters, peer_iters+size);
    max_it = *std::max_element(peer_iters, peer_iters+size);

//    if (rank == 0) {
//        for (int i=0; i<size; i++) {
//
//            printf("pre replay: Peer iters [%d] = %d\n", i, peer_iters[i]);
//
//        }
//
//    }

    // only survivors replay
    if (rank != last_dead) {

        int ew_size = P;
        int ns_size = size/P;
        int ew_rank = rank%P;
        int ns_rank = rank/P;
        for (int i = min_it; i<peer_iters[rank]; i++) {
            
            //down_up(comm, i, rank, size); // EXTREMELY IMPORTANT TO AVOID DEADLOCKS !!!
            MPI_Request req[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
            if( 0 != ns_rank )  {
                send_wrapper(NULL, NB, MPI_TYPE, rank - P, 0, comm, &req[0], i, 0);
            }
            if( (ns_size-1) != ns_rank ) {
                send_wrapper(NULL, NB, MPI_TYPE, rank + P, 0, comm, &req[1], i, 1);
            }

            if( (ew_size-1) != ew_rank ) {
                send_wrapper(NULL, MB, MPI_TYPE, rank + 1, 0, comm, &req[2], i, 2);
            }
            if( 0 != ew_rank ) {
                send_wrapper(NULL, MB, MPI_TYPE, rank - 1, 0, comm, &req[3], i, 3);
            }
            //printf("Rank %d: replay iter %d starting waitall\n", rank, i);
            MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

            //printf("Rank %d: replay iter %d finishing waitall\n", rank, i);
            int tmp_min_it = *std::min_element(peer_iters,peer_iters+size);
            //int tmp_max_it = *std::min_element(peer_iters,peer_iters+size);
            for (int i1=0; i1<size; i1++) 
                if (tmp_min_it == peer_iters[i1]) peer_iters[i1]++;

            //for (int i1=0; i1<size; i1++) {
            //    printf("Rank %d - replay it %d, peer_iters[%d] = %d\n", rank, i, i1, peer_iters[i1]);
            //}
            //printf("Rank %d: finished with replay iter = %d, ns_rank = %d, ew_rank = %d, ns_size = %d, ew_size - %d\n", rank, i, ns_rank, ew_rank, ns_size, ew_size);

        }
    }
    if (rank == 0) {
        for (int i=0; i<size; i++) {

            printf("post replay: Peer iters [%d] = %d\n", i, peer_iters[i]);

        }

    }

    //printf("Rank %d: Replay routine finishing \n", rank);

}

int send_wrapper(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
        MPI_Comm comm, MPI_Request * request, int replay_it, int stage) {

    //return MPI_Isend(buf, count, datatype, dest, tag, comm, request);
            //return MPI_Isend(buf, count, datatype, dest, tag, comm, request);
    //printf("Rank %d: in send wrapper, dest = %d, iter = %d-%d, dest iter = %d\n", rank, dest, replay_it, stage, peer_iters[dest]);

    if (replay_it == peer_iters[dest]) {
        auto p = std::make_pair(dest, std::make_pair(replay_it % CKPT_STEP,stage));
        // regular send
        if (peer_iters[rank] == peer_iters[dest]) {
            int dt_size;
            MPI_Type_size(datatype, &dt_size);
            // allocate if needed
            if (logs.find(p) == logs.end()) {
                logs[p] = malloc(dt_size * count);
            }
            memcpy(logs[p], buf, dt_size * count);
            //append_log(replay_it, stage, buf, count);
            //printf("Rank %d: normal send to %d in (%d-%d)\n", rank, dest, replay_it, stage);
            return MPI_Isend(buf, count, datatype, dest, tag, comm, request);
        }
        // replay send
        else if (peer_iters[rank] > peer_iters[dest])  {
            printf("Rank %d: REPLAYING  to %d iter %d-%d\n", rank, dest, replay_it, stage);
            if (logs.find(p) == logs.end()) {
                fprintf(stderr, "Couldn't find a log where it was expected\n");
                MPI_Abort(world, -1);
            }
            void * buf2 = logs[p];
            return MPI_Isend(buf2, count, datatype, dest, tag, comm, request);
        }
        else {
            printf("Rank %d: No match in send wrapper (%d-%d)\n", rank, replay_it, stage);
            return 0;
        }
    }
    else {
        printf("Rank %d: ignores replay to %d in send wrapper (%d-%d). because peer_iters[dest] = %d\n", rank, dest, replay_it, stage, peer_iters[dest]);
        return 0;
    }
}

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
                      int tag, MPI_Comm comm, MPI_Request *request) {

    //printf("Rank %d iter %d - call recv from %d\n", rank, iteration, source);
    return PMPI_Irecv(buf,count,datatype,source,tag,comm,request);
}
