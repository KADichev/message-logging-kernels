#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <setjmp.h>
#include <stdbool.h>
#include <signal.h>
#include <vector>
#include <mammut/mammut.hpp>
#include "mammut_functions.hpp"
#include "timers.h"
#include "mpi.h"
#include "mpi-ext.h"

//#define DEBUG_ITER_COUNT

//#define ENABLE_EVENT_LOG_ 0

#ifdef ENABLE_EVENT_LOG_
int EVENT_LOGGER;
#endif

#include <math.h>
#define max(a,b) ((a>b) ? (a) : (b))
#define min(a,b) ((a<b) ? (a) : (b))
static double start_it;
static int max_iter = 0;
static bool post_failure = false;
static bool post_failure_sync = false;
static int FAILED_PROC;
static int KILL_OUTER_ITER;
static int KILL_INNER_ITER;
static int KILL_PHASE;
#define CGITMAX 50
// from S class
//#define NONZER 7
#define NITER 2
//#define SHIFT 10.0
#define RCOND 1.0e-1
//#define NA 1400
//
//from A clas
//
//#define NA 14000
//#define NONZER 11
//#define SHIFT 20
//from B class
//#define NA 75000
//#define NONZER 13
//#define SHIFT 60
//#define NITER 25
//#define RCOND 1.0e-1
//from C class
//#define NA 150000
//#define NITER 75
//#define NONZER 15
//#define SHIFT 110
// from D class
#define NA 1500000
#define NONZER 21
#define SHIFT 500
//E class
//#define NA 9000000
//#define NONZER 26
//#define SHIFT 1500

#define NZ  NA*(NONZER+1)*(NONZER+1)+NA*(NONZER+2)+NONZER

/* common /partit_size/ */
typedef int boolean;
static int naa, nzz, firstrow, lastrow, firstcol, lastcol, send_start, send_len;
bool ap_q, exchange, obtain_sum, obtain_rho;

//static double w[NA+2+1];  /* w[1:NA+2] */
//
///* common /urando/ */
static int dummy, dummy2;
static int allowed_to_kill = 1;
static int killed_times = 0;
static int *last_ckpt;
static int cgit;
static int * peer_iters;
static double ** replay1;
static double *** replay2;
static double ** replay4;
static double * replay5;
static int * replay_ind = 0;
static int outer_iter;
static double amult;
static double tran;
int l2npcols, nprocs, npcols, nprows, exch_recv_length, exch_proc;
int me;
extern double randlc(double *, double);


/* function declarations */
static void conj_grad (int colidx[], int rowstr[], double x[], double z[], double a[], double p[], double q[], double r[], double w[], double *rnorm, int reduce_exch_proc[], int reduce_send_starts[], int reduce_send_lengths[], int reduce_recv_starts[], int reduce_recv_lengths[], MPI_Comm * parent);
static void makea(int n, int nz, double a[], int colidx[], int rowstr[], int nonzer, int firstrow, int lastrow, int firstcol, int lastcol, double rcond, int arow[], int acol[], double aelt[], double v[], int iv[], double shift ); 
static void sparse(double a[], int colidx[], int rowstr[], int n, int arow[], int acol[], double aelt[], int firstrow, int lastrow, double x[], boolean mark[], int nzloc[], int nnza);
static void sprnvc(int n, int nz, double v[], int iv[], int nzloc[], int mark[]);
static int icnvrt(double x, int ipwr2);
static void vecset(int n, double v[], int iv[], int *nzv, int i, double val);
                         


static int MPIX_Comm_replace(MPI_Comm comm, MPI_Comm *newcomm);

int verbose = 1;
//static int rank = MPI_PROC_NULL, verbose = 1; /* makes this global (for printfs) */
static char estr[MPI_MAX_ERROR_STRING]=""; static int strl; /* error messages */

char** gargv;

static int iteration = 0, ckpt_iteration = 0, last_dead = MPI_PROC_NULL;

static jmp_buf stack_jmp_buf;

/* mockup checkpoint restart: we reset iteration, and we prevent further
 * error injection */
static int app_reload_ckpt(MPI_Comm comm)
{
    /* Fall back to the last checkpoint */
    MPI_Allreduce(&ckpt_iteration, &iteration, 1, MPI_INT, MPI_MAX, comm);
    iteration++;
    return 0;
}

/* world will swap between worldc[0] and worldc[1] after each respawn */
static MPI_Comm worldc[2] = { MPI_COMM_NULL, MPI_COMM_NULL };
static int worldi = 0;

#define world (worldc[worldi])


void setup_failures() {

  FILE * f = fopen("failures.cg.conf", "r");
  if (f == NULL) {
    fprintf(stderr, "No failure config file\n");
    MPI_Abort(world, -1);
  }
  fscanf (f, "%d : %d : %d : %d", &FAILED_PROC, &KILL_OUTER_ITER, &KILL_INNER_ITER, &KILL_PHASE);
  fclose(f);
}

/* repair comm world, reload checkpoints, etc...
 *  Return: true: the app needs to redo some iterations
 *          false: no failure was fixed, we do not need to redo any work.
 */
static int app_needs_repair(MPI_Comm comm)
{
    printf("(Rank %d): enter app_needs_repair\n", me);
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

        // ToDo: shouldn't have to do that here?
        //app_reload_ckpt(world);
        if( MPI_COMM_NULL == comm ) return false; /* ok, we repaired nothing, no need to redo any work */
        //do_recover = 1;
        //goto restart;
        longjmp( stack_jmp_buf, 1 );
    }
    return true; /* we have repaired the world, we need to reexecute */
}

/* Do all the magic in the error handler */
static void errhandler_respawn(MPI_Comm* pcomm, int* errcode, ...)
{
  //int rank; MPI_Comm_rank(*pcomm, &rank);
  //printf("DEBUG: rank in respawn -> %d\n", rank);
    int eclass;
    MPI_Error_class(*errcode, &eclass);

    //printf("Rank %d: ap_q = %d, exchange = %d, obtain_sum = %d, obtain_rho = %d\n", me, ap_q, exchange, obtain_sum, obtain_rho);
    if( verbose ) {
        MPI_Error_string(*errcode, estr, &strl);
        fprintf(stderr, "%04d: errhandler invoked in cgit = %d with error %s\n", me, cgit, estr);
    }

    if( MPIX_ERR_PROC_FAILED != eclass &&
        MPIX_ERR_REVOKED != eclass ) {
        MPI_Abort(MPI_COMM_WORLD, *errcode);
    }
    MPIX_Comm_revoke(world);

    timer_start(2);
    app_needs_repair(world);
    timer_stop(2);
}

static int MPIX_Comm_replace(MPI_Comm comm, MPI_Comm *newcomm)
{
    MPI_Comm icomm, /* the intercomm between the spawnees and the old (shrinked) world */
        scomm, /* the local comm for each sides of icomm */
        mcomm; /* the intracomm, merged from icomm */
    MPI_Group cgrp, sgrp, dgrp;
    int rc, flag, rflag, i, crank, srank, drank, nc, ns, nd;
    printf("process %d enters replace ...\n", me);

 redo:
    if( comm == MPI_COMM_NULL ) { /* am I a new process? */
        /* I am a new spawnee, waiting for my new rank assignment
         * it will be sent by rank 0 in the old world */
        MPI_Comm_get_parent(&icomm);
        scomm = MPI_COMM_WORLD;

        MPI_Recv(&crank, 1, MPI_INT, 0, 1, icomm, MPI_STATUS_IGNORE);

        printf("New process should be assigned rank %d\n", crank);

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
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Info_set(info, "host", "kos0");
        rc = MPI_Comm_spawn(gargv[0], &gargv[1], nd, info,
                            0, scomm, &icomm, MPI_ERRCODES_IGNORE);
        flag = (MPI_SUCCESS == rc);
        MPIX_Comm_agree(scomm, &flag);
        if( !flag ) {
            if( MPI_SUCCESS == rc ) {
                MPIX_Comm_revoke(icomm);
                MPI_Comm_free(&icomm);
            }
            MPI_Comm_free(&scomm);
            if( verbose ) fprintf(stderr, "%04d: comm_spawn failed, redo\n", me);
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
        if( verbose ) fprintf(stderr, "%04d: Intercomm_merge failed, redo\n", me);
        goto redo;
    }

    /* Now, reorder mcomm according to original rank ordering in comm
     * Split does the magic: removing spare processes and reordering ranks
     * so that all surviving processes remain at their former place */
    //printf("Trying to assign rank %d in split\n", crank);
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
        if( verbose ) fprintf(stderr, "%04d: comm_split failed, redo\n", me);
        goto redo;
    }

    /* restore the error handler */
    if( MPI_COMM_NULL != comm ) {
        MPI_Errhandler errh;
        MPI_Comm_get_errhandler( comm, &errh );
        MPI_Comm_set_errhandler( *newcomm, errh );
    }
    //printf("Done with the recovery (rank %d)\n", crank);

    return MPI_SUCCESS;
}


void FPrintSparseMatrix (FILE *f, int *vptr, int *vpos, double *vval, int dim1)
{
  int i, j;

  if (vptr == vpos)
  {
    fprintf (f, "Diagonals : \n ");
    for (i=1; i<=dim1; i++) fprintf (f, "%f ", vval[i]); fprintf (f, "\n");
  }

  fprintf (f, "Pointers: \n ");
  if (dim1 > 0)
    for (i=1; i<=dim1+1; i++) fprintf (f, "%d ", vptr[i]); fprintf (f, "\n");

  fprintf (f, "Values: \n");
  for (i=1; i<=dim1; i++) {
     fprintf (f, " Row %d --> ", i);

    for (j=(vptr[i]); j<(vptr[i+1]); j++)
      fprintf (f, "(%d,%lf) ", vpos[j+1], vval[j+1]); 
    fprintf (f, "\n");  
  }
  fprintf (f, "\n");
}

void FReadSparseMatrix (FILE *f, int *vptr, int *vpos, double *vval, int dim1)
{
  int i, j;

  if (vptr == vpos)
  {
    fscanf (f, "Diagonals : \n ");
    for (i=1; i<=dim1; i++) fscanf (f, "%f ", &vval[i]); fscanf (f, "\n");
  }

  fscanf (f, "Pointers: \n ");
  if (dim1 > 0)
    for (i=1; i<=dim1+1; i++) fscanf (f, "%d ", &vptr[i]); fscanf (f, "\n");

  fscanf (f, "Values: \n");
  for (i=1; i<=dim1; i++)
  { int k; 
    fscanf (f, " Row %d --> ", &k);
    for (j=(vptr[i]); j<(vptr[i+1]); j++) {
      fscanf (f, "(%d,%lf) ", &vpos[j+1], &vval[j+1]); 
    }
    fscanf (f, "\n");  
  }
  fscanf (f, "\n");
}

void writevec(double z[], double p[], double r[], double q[], double rho) {

    //don't measure I/O
    timer_start(1);
   char filename[16];
   int dim1 = lastcol-firstcol+1;
   sprintf (filename, "/tmp/x-%d.dat", me);
   FILE * file = fopen(filename, "w");
   if (file == NULL) {
     printf("Can't write backup file\n");
     MPI_Abort(world, -1);
   }
   fprintf (file, "(%d,%d,%12.16e)\n", outer_iter, cgit, rho);
   for (int i=1; i<=dim1; i++)
     fprintf (file, "(%12.16e,%12.16e,%12.16e,%12.16e) ", z[i], p[i], r[i], q[i]);
   fprintf (file, "\n");
   fclose(file);
  timer_stop(1);
}

void readvec (double z[], double p[], double r[], double q[], double *rho) {

    //don't measure I/O
   timer_start(1);
   char filename[16];
   int dim1 = lastcol-firstcol+1;
   sprintf (filename, "/tmp/x-%d.dat", me);
   FILE * file = fopen(filename, "r");
   fscanf (file, "(%d,%d,%lf)\n", &outer_iter, &cgit, rho);
   //printf("Just read rho = %12.12e\n", rho);
   for (int i=1; i<=dim1; i++) {
     fscanf (file, "(%lf,%lf,%lf,%lf) ", &z[i], &p[i], &r[i], &q[i]);
     //printf("after READ x[%d] = %lf\n", colidx[i], x[colidx[i]]);
   }
   fscanf (file, "\n");
   fclose(file);
   timer_stop(1);
}

void writea(int rowstr[], int colidx[], double a[]) {

   char filename[16];
   sprintf (filename, "a-%d.dat", me);
   FILE * file = fopen(filename, "w");
   FPrintSparseMatrix(file, rowstr, colidx, a, lastcol-firstcol+1);
  // dummy test for Fotran style in C data structures
  // double  a1[] = {0, 5,8,3,6};
  // int rowstr1[] = {0,0,0,2,3,4};
  // int colidx1[] = {0,1,2,3,2};
  // FPrintSparseMatrix(file, rowstr1, colidx1, a1, 4);
  fclose(file);
}

void reada(int rowstr[], int colidx[], double a[]) {

  char filename[16];
  sprintf (filename, "matrix-%d.dat", me);
  FILE * file = fopen(filename, "r");
  if (file == NULL) {
    printf("Can't read matrix file\n");
    MPI_Abort(world, -1);
  }
  FReadSparseMatrix(file, rowstr, colidx, a, lastcol-firstcol+1);
  fclose(file);
}


/*---------------------------------------------------------------------
c       generate the test problem for benchmark 6
c       makea generates a sparse matrix with a
c       prescribed sparsity distribution
c
c       parameter    type        usage
c
c       input
c
c       n            i           number of cols/rows of matrix
c       nz           i           nonzeros as declared array size
c       rcond        r*8         condition number
c       shift        r*8         main diagonal shift
c
c       output
c
c       a            r*8         array for nonzeros
c       colidx       i           col indices
c       rowstr       i           row pointers
c
c       workspace
c
c       iv, arow, acol i
c       v, aelt        r*8
c---------------------------------------------------------------------*/
static void makea(
    int n,
    int nz,
    double a[],		/* a[1:nz] */
    int colidx[],	/* colidx[1:nz] */
    int rowstr[],	/* rowstr[1:n+1] */
    int nonzer,
    int firstrow,
    int lastrow,
    int firstcol,
    int lastcol,
    double rcond,
    int arow[],		/* arow[1:nz] */
    int acol[],		/* acol[1:nz] */
    double aelt[],	/* aelt[1:nz] */
    double v[],		/* v[1:n+1] */
    int iv[],		/* iv[1:2*n+1] */
    double shift )
{
    int i, nnza, iouter, ivelt, ivelt1, irow, nzv;

/*--------------------------------------------------------------------
c      nonzer is approximately  (int(sqrt(nnza /n)));
c-------------------------------------------------------------------*/

    double size, ratio, scale;
    int jcol;

    size = 1.0;
    ratio = pow(rcond, (1.0 / (double)n));
    nnza = 0;

/*---------------------------------------------------------------------
c  Initialize colidx(n+1 .. 2n) to zero.
c  Used by sprnvc to mark nonzero positions
c---------------------------------------------------------------------*/
    for (i = 1; i <= n; i++) {
      iv[n+i] = 0;
    }
    for (iouter = 1; iouter <= n; iouter++) {
      nzv = nonzer;
      sprnvc(n, nzv, v, colidx, iv, &iv[n]);
      vecset(n, v, colidx, &nzv, iouter, 0.5);
      //printf("nzv=%d\n",nzv);
      for (ivelt = 1; ivelt <= nzv; ivelt++) {
        jcol = colidx[ivelt];
        //printf("jcol = %d\n",jcol);
        if (jcol >= firstcol && jcol <= lastcol) {
          scale = size * v[ivelt];
          for (ivelt1 = 1; ivelt1 <= nzv; ivelt1++) {
            irow = colidx[ivelt1];
            if (irow >= firstrow && irow <= lastrow) {
              nnza = nnza + 1;
              //printf("INCR\n");
              if (nnza > nz) {
                printf("Space for matrix elements exceeded in"
                    " makea\n");
                //printf("nnza, nzmax = %d, %d\n", nnza, nz);
                //printf("iouter = %d\n", iouter);
                exit(1);
              }
              //printf("nnza = %d\n",nnza);
              acol[nnza] = jcol;
              arow[nnza] = irow;
              aelt[nnza] = v[ivelt1] * scale;
            }
          }
        }
      }
      size = size * ratio;
    }

/*---------------------------------------------------------------------
c       ... add the identity * rcond to the generated matrix to bound
c           the smallest eigenvalue from below by rcond
c---------------------------------------------------------------------*/
    for (i = firstrow; i <= lastrow; i++) {
      if (i >= firstcol && i <= lastcol) {
        iouter = n + i;
        nnza = nnza + 1;
        if (nnza > nz) {
          printf("Space for matrix elements exceeded in makea\n");
          printf("nnza, nzmax = %d, %d\n", nnza, nz);
          printf("iouter = %d\n", iouter);
          exit(1);
        }
        acol[nnza] = i;
        arow[nnza] = i;
        aelt[nnza] = rcond - shift;
      }
    }

/*---------------------------------------------------------------------
c       ... make the sparse matrix from list of elements with duplicates
c           (v and iv are used as  workspace)
c---------------------------------------------------------------------*/
    sparse(a, colidx, rowstr, n, arow, acol, aelt,
	   firstrow, lastrow, v, iv, &iv[n], nnza);

    for (int j = 1; j <= lastrow - firstrow + 1; j++) {
      for (int k = rowstr[j]; k < rowstr[j+1]; k++) {
        colidx[k] = colidx[k] - firstcol + 1;
      }
    }
}

/*---------------------------------------------------
c       generate a sparse matrix from a list of
c       [col, row, element] tri
c---------------------------------------------------*/
static void sparse(
    double a[],		/* a[1:*] */
    int colidx[],	/* colidx[1:*] */
    int rowstr[],	/* rowstr[1:*] */
    int n,
    int arow[],		/* arow[1:*] */
    int acol[],		/* acol[1:*] */
    double aelt[],	/* aelt[1:*] */
    int firstrow,
    int lastrow,
    double x[],		/* x[1:n] */
    boolean mark[],	/* mark[1:n] */
    int nzloc[],	/* nzloc[1:n] */
    int nnza)
/*---------------------------------------------------------------------
c       rows range from firstrow to lastrow
c       the rowstr pointers are defined for nrows = lastrow-firstrow+1 values
c---------------------------------------------------------------------*/
{
    int nrows;
    int i, j, jajp1, nza, k, nzrow;
    double xi;

/*--------------------------------------------------------------------
c    how many rows of result
c-------------------------------------------------------------------*/
    nrows = lastrow - firstrow + 1;

/*--------------------------------------------------------------------
c     ...count the number of triples in each row
c-------------------------------------------------------------------*/
    for (j = 1; j <= n; j++) {
      rowstr[j] = 0;
      mark[j] = 0;
    }
    rowstr[n+1] = 0;
    
    for (nza = 1; nza <= nnza; nza++) {
      j = (arow[nza] - firstrow + 1) + 1;
      rowstr[j] = rowstr[j] + 1;
    }

    rowstr[1] = 1;
    for (j = 2; j <= nrows+1; j++) {
      rowstr[j] = rowstr[j] + rowstr[j-1];
    }

/*---------------------------------------------------------------------
c     ... rowstr(j) now is the location of the first nonzero
c           of row j of a
c---------------------------------------------------------------------*/
    
/*--------------------------------------------------------------------
c     ... do a bucket sort of the triples on the row index
c-------------------------------------------------------------------*/
      for (nza = 1; nza <= nnza; nza++) {
        j = arow[nza] - firstrow + 1;
        k = rowstr[j];
        a[k] = aelt[nza];
        colidx[k] = acol[nza];
        rowstr[j] = rowstr[j] + 1;
      }

/*--------------------------------------------------------------------
c       ... rowstr(j) now points to the first element of row j+1
c-------------------------------------------------------------------*/
      for (j = nrows; j >= 1; j--) {
        rowstr[j+1] = rowstr[j];
      }
      rowstr[1] = 1;

/*--------------------------------------------------------------------
c       ... generate the actual output rows by adding elements
c-------------------------------------------------------------------*/
    nza = 0;
    for (i = 1; i <= n; i++) {
      x[i] = 0.0;
      mark[i] = 0;
    }

    jajp1 = rowstr[1];
    for (j = 1; j <= nrows; j++) {
      nzrow = 0;

      /*--------------------------------------------------------------------
        c          ...loop over the jth row of a
        c-------------------------------------------------------------------*/
      for (k = jajp1; k < rowstr[j+1]; k++) {
        i = colidx[k];
        x[i] = x[i] + a[k];
        if ( mark[i] == 0 && x[i] != 0.0) {
          mark[i] = 1;
          nzrow = nzrow + 1;
          nzloc[nzrow] = i;
        }
      }

      /*--------------------------------------------------------------------
        c          ... extract the nonzeros of this row
        c-------------------------------------------------------------------*/
      for (k = 1; k <= nzrow; k++) {
        i = nzloc[k];
        mark[i] = 0;
        xi = x[i];
        x[i] = 0.0;
        if (xi != 0.0) {
          nza = nza + 1;
          a[nza] = xi;
          colidx[nza] = i;
        }
      }
      jajp1 = rowstr[j+1];
      rowstr[j+1] = nza + rowstr[1];
    }
}

/*---------------------------------------------------------------------
c       generate a sparse n-vector (v, iv)
c       having nzv nonzeros
c
c       mark(i) is set to 1 if position i is nonzero.
c       mark is all zero on entry and is reset to all zero before exit
c       this corrects a performance bug found by John G. Lewis, caused by
c       reinitialization of mark on every one of the n calls to sprnvc
---------------------------------------------------------------------*/
static void sprnvc(
    int n,
    int nz,
    double v[],		/* v[1:*] */
    int iv[],		/* iv[1:*] */
    int nzloc[],	/* nzloc[1:n] */
    int mark[] ) 	/* mark[1:n] */
{
  int nn1;
  int nzrow, nzv, ii, i;
  double vecelt, vecloc;

  nzv = 0;
  nzrow = 0;
  nn1 = 1;
  do {
    nn1 = 2 * nn1;
  } while (nn1 < n);

  /*--------------------------------------------------------------------
    c    nn1 is the smallest power of two not less than n
    c-------------------------------------------------------------------*/

  while (nzv < nz) {
    vecelt = randlc(&tran, amult);

    /*--------------------------------------------------------------------
      c   generate an integer between 1 and n in a portable manner
      c-------------------------------------------------------------------*/
    vecloc = randlc(&tran, amult);
    i = icnvrt(vecloc, nn1) + 1;
    if (i > n) continue;

    /*--------------------------------------------------------------------
      c  was this integer generated already?
      c-------------------------------------------------------------------*/
    if (mark[i] == 0) {
      mark[i] = 1;
      nzrow = nzrow + 1;
      nzloc[nzrow] = i;
      nzv = nzv + 1;
      v[nzv] = vecelt;
      iv[nzv] = i;
    }
  }

  for (ii = 1; ii <= nzrow; ii++) {
    i = nzloc[ii];
    mark[i] = 0;
  }
}

/*---------------------------------------------------------------------
* scale a double precision number x in (0,1) by a power of 2 and chop it
*---------------------------------------------------------------------*/
static int icnvrt(double x, int ipwr2) {
    return ((int)(ipwr2 * x));
}

/*--------------------------------------------------------------------
c       set ith element of sparse vector (v, iv) with
c       nzv nonzeros to val
c-------------------------------------------------------------------*/
static void vecset(
    int n,
    double v[],	/* v[1:*] */
    int iv[],	/* iv[1:*] */
    int *nzv,
    int i,
    double val)
{
  int k;
  boolean set;

  set = 0;
  for (k = 1; k <= *nzv; k++) {
    if (iv[k] == i) {
      v[k] = val;
      set  = 1;
    }
  }
  if (!set) {
    *nzv = *nzv + 1;
    v[*nzv] = val;
    iv[*nzv] = i;
  }
}

void setup_proc_info(int num_procs) {
//c---------------------------------------------------------------------
//c  num_procs must be a power of 2, and num_procs=num_proc_cols*num_proc_rows
//c  When num_procs is not square, then num_proc_cols = 2*num_proc_rows
//c---------------------------------------------------------------------
//c  First, number of procs must be power of two. 
//c---------------------------------------------------------------------
 //     if( nprocs != num_procs ) {
 //       printf("Number of proceses not same as compiled\n");
 //         MPI_Abort(world, -1);
 //     }


       int root_nprocs = sqrt(num_procs);
      //int lognprocs = log2(num_procs);
      // cover case 1 MPI proc only
      //if (lognprocs == 0) lognprocs = 1;
      if (root_nprocs * root_nprocs != num_procs) {
        printf("Proc number not power of two, not dealing with this ...\n");
        MPI_Abort(world, -1);
      }

      npcols = sqrt(num_procs); //lognprocs;
      nprows = sqrt(num_procs); //lognprocs;
      //printf("(Rank %d): npcols %d, nprows %d\n", me, npcols, nprows);
     
}

void setup_submatrix_info(int * reduce_exch_proc, int * reduce_send_starts, int * reduce_send_lengths, int * reduce_recv_starts, int * reduce_recv_lengths) {

  int proc_row = me / npcols;
  int proc_col = me - proc_row*npcols;

  int col_size, row_size;

  // naa evenly distributed
  if( (naa/npcols)*npcols == naa ) {
    col_size = naa/npcols;
    firstcol = proc_col*col_size + 1;
    lastcol  = firstcol - 1 + col_size;
    row_size = naa/nprows;
    firstrow = proc_row*row_size + 1;
    lastrow  = firstrow - 1 + row_size;
  }
  // not evenly distributed
  else {
    if( proc_row < naa - (naa/nprows)*nprows){
      row_size = naa/nprows+ 1;
      firstrow = proc_row*row_size + 1;
      lastrow  = firstrow - 1 + row_size;
    }
    else {
      row_size = naa/nprows;
      firstrow = (naa - naa/nprows*nprows)*(row_size+1)
        + (proc_row-(naa-naa/nprows*nprows))
        *row_size + 1;
      lastrow  = firstrow - 1 + row_size;
    }
    if( npcols == nprows ) {
      if( proc_col < naa - (naa/npcols)*npcols) {
        col_size = naa/npcols+ 1;
        firstcol = proc_col*col_size + 1;
        lastcol  = firstcol - 1 + col_size;
      }
      else {
        col_size = naa/npcols;
        firstcol = (naa - naa/npcols*npcols)*(col_size+1)
          + (proc_col-(naa-naa/npcols*npcols))
          *col_size + 1;
        lastcol  = firstcol - 1 + col_size;
      }
    }
    else {
      if ((proc_col/2) < naa - naa/(npcols/2)*(npcols/2) ) {
        col_size = naa/(npcols/2) + 1;
        firstcol = (proc_col/2)*col_size + 1;
        lastcol  = firstcol - 1 + col_size;
      }
      else {
        col_size = naa/(npcols/2);
        firstcol = (naa - naa/(npcols/2)*(npcols/2)) *(col_size+1)
          + ((proc_col/2)-(naa-naa/(npcols/2)*(npcols/2)))
          *col_size + 1;
        lastcol  = firstcol - 1 + col_size;
      }
      if( me % 2 == 0 ) {
        lastcol  = firstcol - 1 + (col_size-1)/2 + 1;
      }
      else {
        firstcol = firstcol + (col_size-1)/2 + 1;
        lastcol  = firstcol - 1 + col_size/2;
      }
    }
  }


  if( npcols == nprows ) {
    send_start = 1;
    send_len   = lastrow - firstrow + 1;
  }
  else {
    if( me % 2 == 0 ) {
      send_start = 1;
      send_len   = (1 + lastrow-firstrow+1)/2;
    }
    else {
      send_start = (lastrow-firstrow)/2+1;
      send_len   = (lastrow-firstrow+1)/2;
    }
  }


  // c---------------------------------------------------------------------
  // c  Transpose exchange processor
  // c---------------------------------------------------------------------

  if( npcols == nprows ) 
    exch_proc = (me % nprows )*nprows + me/nprows;
  else
    exch_proc = 2*((me/2 % nprows )*nprows + me/2/nprows) + (me % 2);

  //printf("(Rank %d): in setup, nprows=%d, npcols=%d, exch_proc = %d\n", me, nprows, npcols, exch_proc);

  //c---------------------------------------------------------------------
  //c  Set up the reduce phase schedules...
  //c---------------------------------------------------------------------



      int i = npcols / 2;
      l2npcols = 0;
      while( i > 0 ) {
         l2npcols++;
         i /= 2;
      }
  int div_factor = npcols;
  int j;
  for (int i=0; i<l2npcols; i++) {

    j = ((proc_col+div_factor/2) % div_factor)     + proc_col / div_factor * div_factor;
    reduce_exch_proc[i] = proc_row*npcols + j ;
    //printf("rank %d, reduce_exch_proc[%d] = %d\n", me, i, reduce_exch_proc[i]);
    //fflush(stdout);
    div_factor = div_factor / 2;
  }

  for (int i = l2npcols-1; i>=0; i--) {

    if( nprows == npcols ) {
      reduce_send_starts[i]  = send_start;
      reduce_send_lengths[i] = send_len;
      reduce_recv_lengths[i] = lastrow - firstrow + 1;
    }
    else {
      reduce_recv_lengths[i] = send_len;
      if( i == l2npcols - 1) {
        reduce_send_lengths[i] = (lastrow-firstrow+1-send_len);
        if( me/2*2 ==  me ) 
          reduce_send_starts[i] = send_start + send_len;
        else
          reduce_send_starts[i] = 1;
      }
      else
      {
        reduce_send_lengths[i] = send_len;
        reduce_send_starts[i]  = send_start;
      }
    }
    reduce_recv_starts[i] = send_start;
  }
  exch_recv_length = lastcol - firstcol + 1;

}

void replay(double x[], double z[], double p[], double q[], double r[], double w[],  bool failed, int reduce_exch_proc[], int reduce_send_starts[], int reduce_send_lengths[], double * rho) {
    
    //printf("Rank %d: enter replay\n", me);
    

    // All get the others' iteration number
    int size;
    MPI_Comm_size( world, &size); 


    killed_times++;
    timer_start(3);
    MPI_Allreduce(MPI_IN_PLACE, &killed_times, 1, MPI_INT, MPI_MAX, world);
    timer_stop(3);
    if (killed_times == 1) allowed_to_kill = 0;

    if (failed)  {
        readvec(z, p, r, q, rho);
        cgit++;
    }

    timer_start(3);
    MPI_Allgather( &cgit, 1, MPI_INT, peer_iters, 1, MPI_INT, world); 
    timer_stop(3);
    if (me == 0) {
        for (int i=0; i<size; i++) {
            printf("Rank %d -> peer_iters[%d] = %d\n", me, i, peer_iters[i]);
        }
    }

    for (int i=0; i<size; i++) {
        max_iter = max(max_iter, peer_iters[i]);
    }
    int min_iter = max_iter;
    for (int i=0; i<size; i++) {
        min_iter = min(min_iter, peer_iters[i]);
    }

    int global = (LOG_BFR_DEPTH == 0) || ((LOG_BFR_DEPTH > 0) && (max_iter % CKPT_STEP > LOG_BFR_DEPTH));

    if (failed) {
        if (global) {
            for (int i=0; i<size; i++) peer_iters[i] = cgit;
        }
    }

    if (!failed) {
        // global rollback
        if (global) {
            printf("INFO: Survivor %d will support global rollback\n", me);
            readvec(z, p, r, q, rho);
            cgit++;
            for (int i=0; i<size; i++) peer_iters[i] = cgit;
        }
        // local rollback
        else {
            printf("INFO: Survivor %d will support local rollback\n", me);


            for (int j=min_iter; j<peer_iters[me]; j++) {
                //---------------------------------------------------------------------
                //  Sum the partition submatrix-vec A.p's across rows
                //  Exchange and sum piece of w with procs identified in reduce_exch_proc
                //---------------------------------------------------------------------
                for (int i = l2npcols-1; i>=0; i--) {
                    int peer = reduce_exch_proc[i];
                    if (peer_iters[peer] == j) {
                        //printf("Rank %d: will send replay2 with tag %d to rank %d\n", me, j, peer);
                        timer_start(3);
                        MPI_Send( replay2[i][j%LOG_BFR_DEPTH], reduce_send_lengths[i]+1, MPI_DOUBLE, peer, j, world);
                        timer_stop(3);
                    }
                }

                //---------------------------------------------------------------------
                //  Exchange piece of q with transpose processor:
                //---------------------------------------------------------------------
                if (l2npcols != 0) {
                    if (peer_iters[exch_proc] == j) {
                        //printf("Rank %d: will send replay5 with tag %d to rank %d\n", me, j, exch_proc);
                        *replay_ind = (j % LOG_BFR_DEPTH) * (lastrow-firstrow+1+1);
                        timer_start(3);
                        MPI_Send( &replay5[*replay_ind], send_len+1, MPI_DOUBLE, exch_proc, j, world);
                        timer_stop(3);
                    }
                }

                for (int i=0; i<l2npcols; i++) {
                    int peer = reduce_exch_proc[i];
                    if ( peer_iters[peer] == j){
                        //printf("Rank %d: will send replay4 with tag %d to rank %d\n", me, j, peer);
                        timer_start(3);
                        MPI_Send(  &replay4[i][j%LOG_BFR_DEPTH], 1, MPI_DOUBLE, peer, j, world);
                        timer_stop(3);
                    }
                }

                for (int i = 0; i<l2npcols; i++) {
                    int peer = reduce_exch_proc[i];
                    if (peer_iters[peer] == j) {
                        //printf("Rank %d: will send replay1 with tag %d to rank %d\n", me, j, peer);
                        timer_start(3);
                        MPI_Send( &replay1[i][j%LOG_BFR_DEPTH], 1, MPI_DOUBLE, peer, j, world);
                        timer_stop(3);
                    }
                }

                for (int i=0; i<l2npcols; i++)  {
                    int peer = reduce_exch_proc[i];
                    if (peer_iters[exch_proc] == j) 
                        peer_iters[exch_proc]++;
                    if (peer_iters[peer] == j)
                        peer_iters[peer]++;
                }
            }

            //c---------------------------------------------------------------------
            //c  Obtain rho with a sum-reduce
            //c---------------------------------------------------------------------
        }
    }
    post_failure = true;
    post_failure_sync = true;
    //printf("Rank %d: leaving replay\n", me);
}

void conj_grad (int colidx[], int rowstr[], double x[], double z[], double a[], double p[], double q[], double r[], double w[], double *rnorm, int reduce_exch_proc[], int reduce_send_starts[], int reduce_send_lengths[], int reduce_recv_starts[], int reduce_recv_lengths[], MPI_Comm * parent) {


    std::vector<double> log_joules(CGITMAX);
    std::vector<double> log_times(CGITMAX);
    int size;
    double sum = 0.0;
    double z_tmp[lastcol-firstcol+2];
    double r_tmp[lastcol-firstcol+2];
    MPI_Comm_size( world, &size); 
    if (LOG_BFR_DEPTH > 0) {
        //printf("Allocating replay2 and replay5 PER ITER = %d bytes\n", (l2npcols+1) * (lastrow-firstrow+2) * sizeof(double));
        replay2 = (double ***) malloc(sizeof(double **) * l2npcols);
        for (int i=0; i<l2npcols; i++) {
            replay2[i] = (double **) malloc(sizeof(double *) * LOG_BFR_DEPTH );
            for (int j=0; j<LOG_BFR_DEPTH; j++) {
                replay2[i][j] = (double *) malloc(sizeof(double) * (lastrow-firstrow+2));
            }
        }
        replay4 = (double **) malloc(sizeof(double *) * l2npcols);
        replay1 = (double **) malloc(sizeof(double *) * l2npcols);
        for (int i=0; i<l2npcols; i++) {
            replay4[i] = (double *) malloc(sizeof(double) * LOG_BFR_DEPTH);
            replay1[i] = (double *) malloc(sizeof(double) * LOG_BFR_DEPTH);
        }

        replay5 = (double*)malloc(sizeof(double) * LOG_BFR_DEPTH * (lastrow-firstrow+2));
    }


    peer_iters = (int *) malloc(size*sizeof(int)); 
    for (int i=0; i<size; i++) {
        peer_iters[i] = 0;
    }

    int proc_col, proc_row;
    MPI_Request request;
    MPI_Status status;
    MPI_Errhandler errh;
    MPI_Comm_create_errhandler(&errhandler_respawn, &errh);
    last_ckpt = (int *) malloc(sizeof(int));
    replay_ind = (int *) malloc(sizeof(int));
    cgit = 0;

    int do_recover = 0;
    double rho, alpha, rho0, beta, d;
    MPI_Comm_set_errhandler( world, errh);

//restart:
    
    for (int j=1; j <= naa; j++) {
        q[j] = 0.0;
        z[j] = 0.0;
        r[j] = x[j];
        p[j] = r[j];
        w[j] = 0.0;
    }

#ifdef DEBUG_ITER_COUNT
   char filename2[32];
   sprintf (filename2, "total-iter-%d.dat", me);
   FILE * iter_out = fopen(filename2, "a");
   if (iter_out == NULL) {
     printf("Can't write backup file\n");
     MPI_Abort(world, -1);
   }

#endif

    do_recover = _setjmp(stack_jmp_buf);

    // survivor
    if (do_recover) {
        replay(x, z, p, q, r, w, false, reduce_exch_proc, reduce_send_starts, reduce_send_lengths, &rho);
        goto cg_loop_start;
    }
    // newly spawned after crash
    else if (MPI_COMM_NULL != *parent) {
        replay(x, z, p, q, r, w, true, reduce_exch_proc, reduce_send_starts, reduce_send_lengths, &rho);
        *parent = MPI_COMM_NULL;
        goto cg_loop_start;
    }


    //---------------------------------------------------------------------
    //  rho = r.r
    //  Now, obtain the norm of r: First, sum squares of r elements locally...
    //---------------------------------------------------------------------
    for (int j=1; j<=lastcol-firstcol+1; j++) {
        sum = sum + r[j]*r[j];
    }
    //
    //---------------------------------------------------------------------
    //  Exchange and sum with procs identified in reduce_exch_proc
    //  (This is equivalent to mpi_allreduce.)
    //  Sum the partial sums of rho, leaving rho on all processors
    //---------------------------------------------------------------------


    for (int i = 0; i<l2npcols; i++) {
        MPI_Irecv(&rho, 1, MPI_DOUBLE, reduce_exch_proc[i], 995, world, &request);
        timer_start(3);
        MPI_Send(&sum, 1, MPI_DOUBLE, reduce_exch_proc[i], 995, world);
        MPI_Wait(&request, &status);
        timer_stop(3);
        sum = sum + rho;
    }


    //---------------------------------------------------------------------
    //---->
    //  The conj grad iteration loop
    //---->
    //---------------------------------------------------------------------


    rho = sum;



cg_loop_start:

    for (/* starting point depends on stage*/; cgit <  CGITMAX; cgit++) {
        start_it = MPI_Wtime();
        Config::counter->reset();

        if (post_failure) {
            if (post_failure_sync) {
                if (cgit  == max_iter) {
                    post_failure = false;
                    post_failure_sync = false;
                    down_up(world, cgit, me, size);
                }
            }
        }

        timer_start(8);
        memcpy(z_tmp, z, (lastcol-firstcol+2)*sizeof(double));
        memcpy(r_tmp, r, (lastcol-firstcol+2)*sizeof(double));
        timer_stop(8);

        if (me == 0) printf("(Rank %d): iter = %d-%d\n", me, outer_iter, cgit);
        //
        //---------------------------------------------------------------------
        //  q = A.p
        //  The partition submatrix-vector multiply: use workspace w
        //---------------------------------------------------------------------
ap_q_label:
        ap_q = false;
        for (int j=1; j<=lastrow-firstrow+1; j++) {
            sum = 0.;
            for (int k=rowstr[j]; k<rowstr[j+1]; k++) {
                sum = sum + a[k]*p[colidx[k]];
            }
            w[j] = sum;
        }

        if (outer_iter == KILL_OUTER_ITER && allowed_to_kill && cgit == KILL_INNER_ITER && KILL_PHASE == 1) {
            // No more killing

            if (me == FAILED_PROC) {
                printf("Will kill rank %d ...\n", FAILED_PROC);
                raise(SIGKILL);
            }
        }

        //---------------------------------------------------------------------
        //  Sum the partition submatrix-vec A.p's across rows
        //  Exchange and sum piece of w with procs identified in reduce_exch_proc
        //---------------------------------------------------------------------

        for (int i = l2npcols-1; i>=0; i--) {

            //printf("Rank %d will receive from rank %d in iter = %d-%d\n", me, reduce_exch_proc[i], outer_iter, cgit);
#ifdef EVENT_LOG_
                timer_start(4);
                    MPI_Sendrecv(&dummy, 1, MPI_INT, reduce_exch_proc[0], 0, &dummy2, 1, MPI_INT, reduce_exch_proc[0], 0, world, MPI_STATUS_IGNORE);
                timer_stop(4);
#endif
            MPI_Irecv(&q[reduce_recv_starts[i]-1], reduce_recv_lengths[i]+1, MPI_DOUBLE, reduce_exch_proc[i], cgit, world, &request);
            if (peer_iters[reduce_exch_proc[i]] <= cgit) {

                //printf("Rank %d will send to rank %d, iter = %d-%d\n", me, reduce_exch_proc[i], outer_iter, cgit);
                if ((LOG_BFR_DEPTH > 0) && (cgit % CKPT_STEP < LOG_BFR_DEPTH)) {
                    timer_start(6);
                    memcpy(replay2[i][cgit % LOG_BFR_DEPTH], &w[reduce_send_starts[i]-1], sizeof(double)*(reduce_send_lengths[i]+1));
                    timer_stop(6);
                }
                timer_start(3);
                MPI_Send( &w[reduce_send_starts[i]-1], reduce_send_lengths[i]+1, MPI_DOUBLE, reduce_exch_proc[i], cgit, world);
                timer_stop(3);
            }
            else {}
            timer_start(3);
            MPI_Wait( &request, &status);
            timer_stop(3);
            //printf("Rank %d: received from rank %d in iter = %d-%d\n", me, reduce_exch_proc[i], outer_iter, cgit);

            for (int j=send_start; j <= (send_start+reduce_recv_lengths[i]-1); j++) {
                w[j] += q[j];
            }
        }

        //---------------------------------------------------------------------
        //  Exchange piece of q with transpose processor:
        //---------------------------------------------------------------------

        ap_q = true;


exchange_label:
        exchange = false;
        if( l2npcols != 0) {
#ifdef EVENT_LOG_
                timer_start(4);
                    MPI_Sendrecv(&dummy, 1, MPI_INT, reduce_exch_proc[0], 0, &dummy2, 1, MPI_INT, reduce_exch_proc[0], 0, world, MPI_STATUS_IGNORE);
                timer_stop(4);
#endif
            MPI_Irecv( q, exch_recv_length+1, MPI_DOUBLE, exch_proc, cgit, world, &request);
            if (peer_iters[exch_proc] <= cgit) {
                if ((LOG_BFR_DEPTH > 0) && (cgit % CKPT_STEP < LOG_BFR_DEPTH)) {
                    *replay_ind = (cgit % LOG_BFR_DEPTH) * (lastrow-firstrow+1+1);
                    timer_start(6);
                    memcpy(&replay5[*replay_ind], &w[send_start-1], sizeof(double)*(send_len+1));
                    timer_stop(6);
                }
                timer_start(3);
                MPI_Send( &w[send_start-1], send_len+1, MPI_DOUBLE, exch_proc, cgit, world);
                timer_stop(3);
            }
            else {}
            timer_start(3);
            MPI_Wait( &request, &status);
            timer_stop(3);
        }
        else {
            for (int j=1; j<=exch_recv_length; j++) {
                q[j] = w[j];
            }
        }
        exchange = true;


        //c---------------------------------------------------------------------
        //c  Clear w for reuse...
        //c---------------------------------------------------------------------
        for (int j=1; j<=max(lastrow-firstrow+1, lastcol-firstcol+1 ); j++) {
            w[j] = 0.0;
        }

        //c---------------------------------------------------------------------
        //c  Obtain p.q
        //c---------------------------------------------------------------------
        sum = 0.0;
        for (int j=1;j<=lastcol-firstcol+1;j++) {
            sum += p[j]*q[j];
        }
        if (sum == 0.0) {
            printf("Rank %d: will abort, local sum = p.q == 0\n", me);
            MPI_Abort(world, -1);
        }

        if (outer_iter == KILL_OUTER_ITER && allowed_to_kill && cgit == KILL_INNER_ITER && KILL_PHASE == 2) {
            // No more killing

            if (me == FAILED_PROC) {
                printf("Will kill rank %d ...\n", FAILED_PROC);
                raise(SIGKILL);
            }
        }

obtain_sum_label:
        obtain_sum = false;
        //c---------------------------------------------------------------------
        //c  Obtain d with a sum-reduce
        //c---------------------------------------------------------------------
        for (int i = 0; i<l2npcols; i++) {
            //printf("Rank(%d): I will receive from rank %d\n", me, reduce_exch_proc[i]);
#ifdef EVENT_LOG_
                timer_start(4);
                    MPI_Sendrecv(&dummy, 1, MPI_INT, reduce_exch_proc[0], 0, &dummy2, 1, MPI_INT, reduce_exch_proc[0], 0, world, MPI_STATUS_IGNORE);
                timer_stop(4);
#endif
            MPI_Irecv( &d, 1, MPI_DOUBLE, reduce_exch_proc[i], cgit, world, &request);
            if (peer_iters[reduce_exch_proc[i]] <= cgit) {
                if ((LOG_BFR_DEPTH > 0) && (cgit % CKPT_STEP < LOG_BFR_DEPTH)) {
                    replay4[i][cgit % LOG_BFR_DEPTH] = sum;
                }
                timer_start(3);
                MPI_Send(  &sum, 1, MPI_DOUBLE, reduce_exch_proc[i], cgit, world);
                timer_stop(3);
            }
            else {}
            timer_start(3);
            MPI_Wait( &request, &status);
            timer_stop(3);
            sum += d;
        }
        d = sum;
        obtain_sum = true;

        if (outer_iter == KILL_OUTER_ITER && allowed_to_kill && cgit == KILL_INNER_ITER && KILL_PHASE == 3) {
            // No more killing

            if (me == FAILED_PROC) {
                printf("Will kill rank %d ...\n", FAILED_PROC);
                raise(SIGKILL);
            }
        }
        //c---------------------------------------------------------------------
        //c  Obtain alpha = rho / (p.q)
        //c---------------------------------------------------------------------
        alpha = rho / d;

        //c---------------------------------------------------------------------
        //c  Save a temporary of rho
        //c---------------------------------------------------------------------
        rho0 = rho;

        //c---------------------------------------------------------------------
        //c  Obtain z = z + alpha*p
        //c  and    r = r - alpha*q
        //c---------------------------------------------------------------------
        for (int j=1; j<= lastcol-firstcol+1;j++) {
            z_tmp[j] += alpha*p[j];
            r_tmp[j] -=  alpha*q[j];
        }

        //c---------------------------------------------------------------------
        //c  rho = r.r
        //c  Now, obtain the norm of r: First, sum squares of r elements locally...
        //c---------------------------------------------------------------------
        sum = 0.0;
        for (int j=1; j<=lastcol-firstcol+1; j++) {
            sum += r_tmp[j]*r_tmp[j];
        }

obtain_rho_label:
        obtain_rho = false;
        //c---------------------------------------------------------------------
        //c  Obtain rho with a sum-reduce
        //c---------------------------------------------------------------------
        for (int i = 0; i<l2npcols; i++) {
#ifdef EVENT_LOG_
                timer_start(4);
                    MPI_Sendrecv(&dummy, 1, MPI_INT, reduce_exch_proc[0], 0, &dummy2, 1, MPI_INT, reduce_exch_proc[0], 0, world, MPI_STATUS_IGNORE);
                timer_stop(4);
#endif
            MPI_Irecv( &rho, 1, MPI_DOUBLE, reduce_exch_proc[i], cgit, world, &request);
            if (peer_iters[reduce_exch_proc[i]] <= cgit) {
                if ((LOG_BFR_DEPTH > 0) && (cgit % CKPT_STEP < LOG_BFR_DEPTH)) {
                    replay1[i][(cgit) % LOG_BFR_DEPTH] = sum;
                }
                timer_start(3);
                MPI_Send( &sum, 1, MPI_DOUBLE, reduce_exch_proc[i], cgit, world);
                timer_stop(3);
            }
            else {}
            timer_start(3);
            MPI_Wait( &request, &status);
            timer_stop(3);
            sum += rho;
        }
        obtain_rho = true;
        timer_start(8);
        memcpy(z, z_tmp, (lastcol-firstcol+2)*sizeof(double));
        memcpy(r, r_tmp, (lastcol-firstcol+2)*sizeof(double));
        timer_stop(8);


        if (outer_iter == KILL_OUTER_ITER && allowed_to_kill && cgit == KILL_INNER_ITER && KILL_PHASE == 4) {
            // No more killing

            if (me == FAILED_PROC) {
                printf("Will kill rank %d ...\n", FAILED_PROC);
                raise(SIGKILL);
            }
        }
        rho = sum;

        //c---------------------------------------------------------------------
        //c  Obtain beta:
        //c---------------------------------------------------------------------
        beta = rho / rho0;

        //c---------------------------------------------------------------------
        //c  p = r + beta*p
        //c---------------------------------------------------------------------
        sum = 0.;
        for (int j=1; j<= lastcol-firstcol+1; j++) {
            p[j] = r[j] + beta*p[j];
            sum += p[j];
        }

        // PERSISTENT CHECKPOINTS
        if (cgit % CKPT_STEP == 0) {
            writevec(z, p, r, q, rho);
            *last_ckpt = cgit;
        }
        double end_it = MPI_Wtime();
        mammut::energy::Joules joules = Config::counter->getJoules();
        //total_joules += joules;
        int global = (LOG_BFR_DEPTH == 0) || ((LOG_BFR_DEPTH > 0) && (max_iter % CKPT_STEP > LOG_BFR_DEPTH));
        if (outer_iter == KILL_OUTER_ITER) {
            if (!global) {
                log_joules[cgit] = joules;
                log_times[cgit] = end_it - start_it;
            }
            else {
                log_joules[cgit] += joules;
                log_times[cgit] += end_it - start_it;
            }
        }

#ifdef DEBUG_ITER_COUNT
        fprintf (iter_out, "%d-|\n",me);
        fflush(iter_out);
#endif
    } //                             ! end of do cgit=1,cgitmax



#ifdef DEBUG_ITER_COUNT
    fclose(iter_out);
#endif

    // free
    if (LOG_BFR_DEPTH > 0) {
        for (int i=0; i<l2npcols; i++) {
            for (int j=0; j<LOG_BFR_DEPTH; j++) {
                free(replay2[i][j]);
            }
            free(replay2[i]);
        }

        for (int i=0; i<l2npcols; i++) {
            free(replay4[i]);
        }
        free(replay4);
        for (int i=0; i<l2npcols; i++) {
            free(replay1[i]);
        }
        free(replay1);
        free(replay5);
        free(last_ckpt);
    }
    free(peer_iters);

    //c---------------------------------------------------------------------
    //c  Compute residual norm explicitly:  ||r|| = ||x - A.z||
    //c  First, form A.z
    //c  The partition submatrix-vector multiply
    //c---------------------------------------------------------------------
    for (int j=1; j<=lastrow-firstrow+1;j++) {
        sum = 0.;
        for (int k=rowstr[j];k<rowstr[j+1];k++) {
            sum += a[k]*z[colidx[k]];
        }
        w[j] = sum;
    }

    //c---------------------------------------------------------------------
    //c  Sum the partition submatrix-vec A.z's across rows
    //c---------------------------------------------------------------------
    for (int i = l2npcols-1; i>=0; i--) {
        MPI_Irecv( &r[reduce_recv_starts[i]-1], reduce_recv_lengths[i]+1, MPI_DOUBLE, reduce_exch_proc[i], cgit, world, &request);
        MPI_Send( &w[reduce_send_starts[i]-1], reduce_send_lengths[i]+1, MPI_DOUBLE, reduce_exch_proc[i], cgit, world);
        MPI_Wait( &request, &status);

        for (int j=send_start; j<= send_start + reduce_recv_lengths[i] - 1; j++) {
            w[j] += r[j];
        }
    }

    //c---------------------------------------------------------------------
    //c  Exchange piece of q with transpose processor:
    //c---------------------------------------------------------------------
    if( l2npcols != 0 ) {
        MPI_Irecv( &r[send_start-1], exch_recv_length+1, MPI_DOUBLE, exch_proc, cgit, world, &request);
        MPI_Send(  &w[send_start-1], send_len+1, MPI_DOUBLE, exch_proc, cgit, world);
        MPI_Wait( &request, &status);
    }
    else {
        for (int j=1; j<=exch_recv_length; j++) {
            r[j] = w[j];
        }
    }


    //c---------------------------------------------------------------------
    //c  At this point, r contains A.z
    //c---------------------------------------------------------------------
    sum = 0.0;
    for (int j=1; j<=lastcol-firstcol+1; j++) {
        d   = x[j] - r[j];
        sum += d*d;
    }

    //c---------------------------------------------------------------------
    //c  Obtain d with a sum-reduce
    //c---------------------------------------------------------------------
    for (int i = 0; i<l2npcols; i++) {
        MPI_Irecv( &d, 1, MPI_DOUBLE, reduce_exch_proc[i], cgit, world, &request);
        MPI_Send( &sum, 1, MPI_DOUBLE, reduce_exch_proc[i], cgit, world);
        MPI_Wait( &request, &status);
        sum +=  d;
    }
    d = sum;

    int root = 0;
    if( me == root ) *rnorm = sqrt( d );


    if (outer_iter == KILL_OUTER_ITER)
    log_stats(&log_joules[0], &log_times[0], CGITMAX, KILL_INNER_ITER, nprocs, FAILED_PROC, world, me);
}


int main(int argc, char **argv) {

  int i = 0;
  timer_clear(0);
  timer_clear(1);
  timer_clear(2);
  timer_clear(3);
  timer_clear(4);
  timer_clear(6);
  timer_clear(7);

 //if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) exit(1);
  MPI_Init(&argc, &argv);
  init_mammut();

  gargv = argv;
  

  const int niter = NITER;
  naa = NA;
  nzz = NZ;
  double rnorm;

  MPI_Comm parent;
  /* Am I a spare ? */
  MPI_Comm_get_parent( &parent );
  if( MPI_COMM_NULL == parent ) {
      /* First run: Let's create an initial world,
       *          * a copy of MPI_COMM_WORLD */
      MPI_Comm_dup( MPI_COMM_WORLD, &world );
  } else {
      //allowed_to_kill = 0;

      /* I am a spare, lets get the repaired world */
      app_needs_repair(MPI_COMM_NULL);
  }

  MPI_Comm_rank(world, &me);
  MPI_Comm_size(world, &nprocs);

#ifdef ENABLE_EVENT_LOG_
  EVENT_LOGGER = nprocs-1;
  nprocs--;
  if (me == EVENT_LOGGER) {
      printf("Me: %d - event logger\n", me);
      while (true) {
          int b;
          MPI_Status status;
          PMPI_Recv(&b, 1, MPI_INT, MPI_ANY_SOURCE, 4444, world, &status);
          if (b != 1)
              PMPI_Send(&b, 1, MPI_INT, status.MPI_SOURCE, 4444, world);
          else {
              MPI_Finalize();
              return 0;
          }
      }
  }
#endif

  char hostname[64];
  gethostname(hostname,64);
  printf("PROC: %d - HOSTNAME: %s\n", me, hostname);
  setup_proc_info(nprocs);
  setup_failures();

  int reduce_exch_proc[npcols], reduce_send_starts[npcols], reduce_send_lengths[npcols], reduce_recv_starts[npcols], reduce_recv_lengths[npcols];

  setup_submatrix_info(reduce_exch_proc, reduce_send_starts, reduce_send_lengths, reduce_recv_starts, reduce_recv_lengths );
  tran    = 314159265.0;
  amult   = 1220703125.0;
  double zeta    = randlc( &tran, amult );


  int * colidx = (int *) malloc(sizeof(int) *(NZ+1));  /* colidx[1:NZ] */
  int * rowstr = (int *) malloc(sizeof(int) * (NA+1+1));  /* rowstr[1:NA+1] */
  int * iv = (int *) malloc(sizeof(int) * (2*NA+1+1));  /* iv[1:2*NA+1] */
  int * arow = (int *) malloc(sizeof(int) * (NZ+1));    /* arow[1:NZ] */
  int * acol = (int *) malloc(sizeof(int) * (NZ+1));    /* acol[1:NZ] */

  /* common /main_flt_mem/ */
  double * v = (double *) malloc(sizeof(double) *(NA+1+1));   /* v[1:NA+1] */
  double * aelt = (double *) malloc(sizeof(double) * (NZ+1)); /* aelt[1:NZ] */
  double * a = (double *) malloc(sizeof(double) * (NZ+1));    /* a[1:NZ] */
  double * x = (double *) malloc(sizeof(double) * (NA+2+1));  /* x[1:NA+2] */
  double * z = (double *) malloc(sizeof(double) * (NA+2+1));  /* z[1:NA+2] */
  double * p = (double *) malloc(sizeof(double) * (NA+2+1));  /* p[1:NA+2] */
  double * q = (double *) malloc(sizeof(double) * (NA+2+1));  /* q[1:NA+2] */
  double * r = (double *) malloc(sizeof(double) * (NA+2+1));  /* r[1:NA+2] */
  double * w = (double *) malloc(sizeof(double) * (NA+2+1));  /* w[1:NA+2] */
 makea(naa, nzz, a, colidx, rowstr, NONZER,
     firstrow, lastrow, firstcol, lastcol, 
     RCOND, arow, acol, aelt, v, iv, SHIFT);
  int j,k;

//
//  printf("Before colidx compute..\n");

    /*--------------------------------------------------------------------
     * c  set starting vector to (1, 1, .... 1)
     * c-------------------------------------------------------------------*/
    for (i = 1; i <= NA+1; i++) {
      x[i] = 1.0;
    }
    for (j = 1; j <= lastcol-firstcol+1; j++) {
      q[j] = 0.0;
      z[j] = 0.0;
      r[j] = 0.0;
      p[j] = 0.0;
    }
  // end omp parallel

  double norm_temp1[2] = {0.,0.};
  double norm_temp2[2] = {0.,0.};
  for (outer_iter=0; outer_iter < niter; outer_iter++) {

    norm_temp1[0] = 0.;norm_temp1[1] = 0.; norm_temp2[0] = 0.;norm_temp2[1] = 0.;
    timer_start(0);
    conj_grad ( colidx, rowstr, x, z, a,  p, q,  r, w, &rnorm, reduce_exch_proc, reduce_send_starts, reduce_send_lengths, reduce_recv_starts, reduce_recv_lengths, &parent);
    timer_stop(0);
    if (me == 0) printf("CG iters %d has time %lld\n", outer_iter, timer_read(0));


    //c---------------------------------------------------------------------
    //c  zeta = shift + 1/(x.z)
    //c  So, first: (x.z)
    //c  Also, find norm of z
    //c  So, first: (z.z)
    //c---------------------------------------------------------------------
    for (int j=1; j<=lastcol-firstcol+1; j++) {
      norm_temp1[0] += x[j]*z[j];
      norm_temp1[1] += z[j]*z[j];
    }

    MPI_Request request;
    MPI_Status status;
    for (int i = 0; i<l2npcols; i++) {
      MPI_Irecv( norm_temp2, 2, MPI_DOUBLE, reduce_exch_proc[i], 996, world, &request);
      MPI_Send(  norm_temp1, 2, MPI_DOUBLE, reduce_exch_proc[i], 996, world);
      MPI_Wait( &request, &status);

      norm_temp1[0] = norm_temp1[0] + norm_temp2[0];
      norm_temp1[1] = norm_temp1[1] + norm_temp2[1];
    }

    norm_temp1[1] = 1.0/ sqrt( norm_temp1[1] );


    if(me == 0) {
      zeta = SHIFT + 1.0 / norm_temp1[0];
    }

    //c---------------------------------------------------------------------
    //c  Normalize z to obtain x
    //c---------------------------------------------------------------------
    for (int j=1; j<=lastcol-firstcol+1; j++) {
      x[j] = norm_temp1[1]*z[j];
    }

    if (me == 0) {
      printf("(Rank %d:)    %5d       %12.12e %lf\t Event logging time: %lf; CG iter time: %lf; I/O time %lf; memcpy timer: %lf; ULFM repair time %lf MPI time %lf, simulate event logging %12.12e, deep copying = %12.12e\n", me, outer_iter, rnorm, zeta, timer_read(4), timer_read(0), timer_read(1), timer_read(6), timer_read(2), timer_read(3), timer_read(7), timer_read(8));
    }

  }


#ifdef ENABLE_EVENT_LOG_
   if (me == 0) {
       int a = 1;
       MPI_Send(&a, 1, MPI_INT, EVENT_LOGGER, 4444, world);
   }
#endif


  free(colidx);
  free(rowstr);
  free(iv);
  free(arow);
  free(acol);
  free(v);
  free(aelt);
  free(a);
  free(x);
  free(z);
  free(p);
  free(q);
  free(r);
  free(w);
  //writea(rowstr, colidx, a);
  printf("Process %d: before finalize\n", me);
  MPI_Barrier(world);
  MPI_Finalize();
  return 0;
}


#ifdef ENABLE_EVENT_LOG_
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
                      int tag, MPI_Comm comm, MPI_Request * request) {
    int a = 0;
    int b;
    timer_start(7);
    PMPI_Sendrecv(&a, 1, MPI_INT, EVENT_LOGGER, 4444, &b, 1, MPI_INT, EVENT_LOGGER, 4444, world, MPI_STATUS_IGNORE);
    timer_stop(7);
    return PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
}
#endif


#ifdef ENABLE_EVENT_LOG_
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) 
{
    int a = 0;
    int b;
    timer_start(7);
    MPI_Sendrecv(&a, 1, MPI_INT, EVENT_LOGGER, 4444, &b, 1, MPI_INT, EVENT_LOGGER, 4444, world, MPI_STATUS_IGNORE);
    timer_stop(7);
    return PMPI_Send(buf, count, datatype, dest, tag, comm);
}
#endif
