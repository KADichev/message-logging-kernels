
static double start[64], elapsed[64];
static long long start_p[64], elapsed_p[64];

void timer_start(int n);
void timer_clear(int n);
void timer_stop( int n );
double timer_read(int n);
