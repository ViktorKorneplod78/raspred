#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>

#define NI 4
#define NJ 4
#define NK 4

#define BLOCK_SIZE 2

double bench_t_start, bench_t_end;

static
double rtclock()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0)
      fprintf (stderr, "Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start()
{
  bench_t_start = rtclock ();
}

void bench_timer_stop()
{
  bench_t_end = rtclock ();
}

void bench_timer_print()
{
  printf("%0.6lf\n", bench_t_end - bench_t_start);
}

static
void init_array(int ni, int nj, int nk,
                float *alpha,
                float *beta,
                float C[ ni][nj],
                float A[ ni][nk],
                float B[ nk][nj],
                int block_size,
                int rank,
                int size)
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;

  // Инициализация A и B только на процессе 0
  if (rank == 0) {
    for (i = 0; i < ni; i++) {
      for (j = 0; j < nk; j++)
        A[i][j] = (float) (i*(j+1) % nk) / nk;
    }

    for (i = 0; i < nk; i++)
      for (j = 0; j < nj; j++)
        B[i][j] = (float) (i*(j+2) % nj) / nj;
  }

  // Инициализация C: сначала везде 0, затем только в блоках, принадлежащих текущему процессу
  int num_blocks_i = (ni + block_size - 1) / block_size;
  int num_blocks_j = (nj + block_size - 1) / block_size;

  // Сначала установим всю матрицу C в 0
  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++) {
      C[i][j] = 0.0;
    }
  }

  // Теперь инициализируем только блоки, принадлежащие текущему процессу
  for (int ii = 0; ii < ni; ii += block_size) {
    for (int jj = 0; jj < nj; jj += block_size) {
      int block_id = ( (ii / block_size) * num_blocks_j ) + (jj / block_size);
      if (block_id % size != rank) continue;

      int i_end = (ii + block_size > ni) ? ni : ii + block_size;
      int j_end = (jj + block_size > nj) ? nj : jj + block_size;

      for (int i = ii; i < i_end; i++) {
        for (int j = jj; j < j_end; j++) {
          C[i][j] = (float) ((i*j+1) % ni) / ni;
        }
      }
    }
  }
}

static
void print_array(int ni, int nj, float C[ ni][nj])
{
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) {
        printf ("%0.2f ", C[i][j]);
    }
    printf("\n");
  }
}

static
void kernel_gemm_blocked(int ni, int nj, int nk,
    float alpha,
    float beta,
    float C[ni][nj],
    float A[ni][nk],
    float B[nk][nj],
    int block_size,
    int rank,
    int size)
{

    int ii, jj, kk;
    int num_blocks_i = (ni + block_size - 1) / block_size;
    int num_blocks_j = (nj + block_size -1) / block_size;

    // Assign blocks to each process
    for (ii = 0; ii < ni; ii += block_size) {
        for (jj = 0; jj < nj; jj += block_size) {
            int block_id = ( (ii / block_size) * num_blocks_j ) + (jj / block_size);
            if (block_id % size != rank) continue;

            int i_end = (ii + block_size > ni) ? ni : ii + block_size;
            int j_end = (jj + block_size > nj) ? nj : jj + block_size;

            // operation
            for (int i = ii; i < i_end; i++) {
                for (int j = jj; j < j_end; j++) {
                    C[i][j] *= beta;
                }
            }

            for (kk = 0; kk < nk; kk += block_size) {
                int k_end = (kk + block_size > nk) ? nk : kk + block_size;
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        for (int k = kk; k < k_end; k++) {
                            C[i][j] += alpha * A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  float alpha;
  float beta;
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  float (*C)[ni][nj]; C = (float(*)[ni][nj])malloc ((ni) * (nj) * sizeof(float));
  float (*A)[ni][nk]; A = (float(*)[ni][nk])malloc ((ni) * (nk) * sizeof(float));
  float (*B)[nk][nj]; B = (float(*)[nk][nj])malloc ((nk) * (nj) * sizeof(float));

  init_array (ni, nj, nk, &alpha, &beta,
       *C,
       *A,
       *B,
       BLOCK_SIZE, rank, size);

  // Casting arrays to all processes
  MPI_Bcast(*A, ni*nk, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(*B, nk*nj, MPI_FLOAT, 0, MPI_COMM_WORLD);

  bench_timer_start();

  kernel_gemm_blocked(ni, nj, nk,
        alpha, beta,
        *C,
        *A,
        *B,
        BLOCK_SIZE, rank, size);

  // Gathering result to process 0
  if (rank == 0) {
      MPI_Reduce(MPI_IN_PLACE, *C, ni*nj, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
      MPI_Reduce(*C, NULL, ni*nj, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  }

  bench_timer_stop();

  if (rank == 0) {
      printf("\n\n");
      printf("Time: ");
      bench_timer_print();

      //if (argc > 42 && ! strcmp(argv[0], ""))
      //printf("\nMATRIX C AFTER CALCULATION:\n");
      //print_array(ni, nj, *C);
  }

  free((void*)C);
  free((void*)A);
  free((void*)B);

  MPI_Finalize();
  return 0;
}
