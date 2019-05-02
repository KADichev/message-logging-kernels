#include "timers.h"
#include "mpi.h"
/*****************************************************************/
/******            T  I  M  E  R  _  C  L  E  A  R          ******/
/*****************************************************************/

void timer_clear(int n) {
        elapsed[n] = 0.0;
}

/*****************************************************************/
/******            T  I  M  E  R  _  S  T  A  R  T          ******/
/*****************************************************************/
//void papi_start( int n ) 
//{
//    start_p[n] = PAPI_get_virt_usec();
//}

void timer_start(int n) {
    start[n] = MPI_Wtime();
}


/*****************************************************************/
/******            T  I  M  E  R  _  S  T  O  P             ******/
/*****************************************************************/
void timer_stop( int n ) 
{
        double t, now;
        now = MPI_Wtime();
        t = now - start[n];
        elapsed[n] += t;
}


//void papi_stop(int n) {
//    long long now = PAPI_get_virt_usec();
//    long long t = now  - start[n];
//    elapsed_p[n] += t;
//}


/*****************************************************************/
/******            T  I  M  E  R  _  R  E  A  D             ******/
/*****************************************************************/
//long long papi_read( int n ) 
//{
//        return elapsed_p[n];
//}

double timer_read(int n) {
        return( elapsed[n] );
}
