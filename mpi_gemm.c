#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>

#define NI 4
#define NJ 4
#define NK 4

#define BLOCK_SIZE 2

double bench_t_start, bench_t_end;

// Структура для хранения информации о блоке (с границами)
typedef struct {
    int start_i;   // Начальная строка блока
    int start_j;   // Начальный столбец блока
    int end_i;     // Конечная строка блока (не включая)
    int end_j;     // Конечный столбец блока (не включая)
    int block_id;  // Идентификатор блока (опционально, для отладки)
} BlockInfo;

// Функция для определения принадлежности блока процессу
static int block_belongs_to_process(int block_i, int block_j, 
                                    int num_blocks_j, int rank, int size) {
    int block_id = block_i * num_blocks_j + block_j;
    return (block_id % size == rank);
}

// Функция для вычисления списка блоков процесса с предварительно вычисленными границами
static BlockInfo* get_process_blocks(int ni, int nj,
                                     int rank, int size,
                                     int *num_blocks)
{
    int num_blocks_i = (ni + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_j = (nj + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Сначала считаем, сколько блоков у процесса
    int count = 0;
    for (int block_i = 0; block_i < num_blocks_i; block_i++) {
        for (int block_j = 0; block_j < num_blocks_j; block_j++) {
            if (block_belongs_to_process(block_i, block_j, num_blocks_j, rank, size)) {
                count++;
            }
        }
    }
    
    *num_blocks = count;
    if (count == 0) return NULL;
    
    // Выделяем память и заполняем
    BlockInfo *blocks = (BlockInfo*)malloc(count * sizeof(BlockInfo));
    
    int idx = 0;
    for (int block_i = 0; block_i < num_blocks_i; block_i++) {
        for (int block_j = 0; block_j < num_blocks_j; block_j++) {
            if (block_belongs_to_process(block_i, block_j, num_blocks_j, rank, size)) {
                blocks[idx].start_i = block_i * BLOCK_SIZE;
                blocks[idx].start_j = block_j * BLOCK_SIZE;
                
                // Вычисляем границы
                blocks[idx].end_i = (blocks[idx].start_i + BLOCK_SIZE > ni)
                                    ? ni : blocks[idx].start_i + BLOCK_SIZE;
                blocks[idx].end_j = (blocks[idx].start_j + BLOCK_SIZE > nj)
                                    ? nj : blocks[idx].start_j + BLOCK_SIZE;
                
                blocks[idx].block_id = block_i * num_blocks_j + block_j;
                idx++;
            }
        }
    }
    
    return blocks;
}

// Функция для печати информации о блоках (для отладки)
static void print_process_blocks_info(BlockInfo *blocks, int num_blocks, 
                                      int rank, int size) {
    fprintf(stderr, "Process %d/%d has %d blocks:\n", rank, size, num_blocks);
    for (int b = 0; b < num_blocks; b++) {
        fprintf(stderr, "  Block %d: rows [%d,%d), cols [%d,%d) [ID=%d]\n",
               b, blocks[b].start_i, blocks[b].end_i, 
               blocks[b].start_j, blocks[b].end_j, blocks[b].block_id);
    }
}

// Функция инициализации массивов
static void init_array_with_blocks(int ni, int nj, int nk,
                                   float *alpha, float *beta,
                                   float C[ni][nj],
                                   float A[ni][nk],
                                   float B[nk][nj],
                                   BlockInfo *blocks, int num_blocks,
                                   int rank) {
    *alpha = 1.5;
    *beta = 1.2;
    
    // Инициализация A и B только на процессе 0
    if (rank == 0) {
        for (int i = 0; i < ni; i++) {
            for (int j = 0; j < nk; j++) {
                A[i][j] = (float) (i*(j+1) % nk) / nk;
            }
        }
        for (int i = 0; i < nk; i++) {
            for (int j = 0; j < nj; j++) {
                B[i][j] = (float) (i*(j+2) % nj) / nj;
            }
        }
    }
    
    // Сначала все элементы C = 0
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            C[i][j] = 0.0;
        }
    }
    
    // Инициализация только своих блоков (уже с известными границами)
    for (int b = 0; b < num_blocks; b++) {
        for (int i = blocks[b].start_i; i < blocks[b].end_i; i++) {
            for (int j = blocks[b].start_j; j < blocks[b].end_j; j++) {
                C[i][j] = (float) ((i*j+1) % ni) / ni;
            }
        }
    }
}

// Функция вычисления GEMM
static void kernel_gemm_with_blocks(int ni, int nj, int nk,
                                    float alpha, float beta,
                                    float C[ni][nj],
                                    float A[ni][nk],
                                    float B[nk][nj],
                                    BlockInfo *blocks, int num_blocks)
{
    for (int b = 0; b < num_blocks; b++) {
        int start_i = blocks[b].start_i;
        int start_j = blocks[b].start_j;
        int end_i = blocks[b].end_i;
        int end_j = blocks[b].end_j;
        
        // Шаг 1: Умножение на beta
        for (int i = start_i; i < end_i; i++) {
            for (int j = start_j; j < end_j; j++) {
                C[i][j] *= beta;
            }
        }
        
        // Шаг 2: Блочное умножение по k
        for (int kk = 0; kk < nk; kk += BLOCK_SIZE) {
            int k_end = (kk + BLOCK_SIZE > nk) ? nk : kk + BLOCK_SIZE;
            
            for (int i = start_i; i < end_i; i++) {
                for (int j = start_j; j < end_j; j++) {
                    for (int k = kk; k < k_end; k++) {
                        C[i][j] += alpha * A[i][k] * B[k][j];
                    }
                }
            }
        }
    }
}

static double rtclock() {
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, NULL);
    if (stat != 0)
      fprintf(stderr, "Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start() {
  bench_t_start = rtclock();
}

void bench_timer_stop() {
  bench_t_end = rtclock();
}

void bench_timer_print() {
  printf("%0.6lf\n", bench_t_end - bench_t_start);
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

int main(int argc, char** argv) {
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  float alpha;
  float beta;
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Получаем список блоков для текущего процесса
  int num_blocks;
  BlockInfo *blocks = get_process_blocks(ni, nj, rank, size, &num_blocks);
  
  // Опционально: печать информации о блоках (для отладки)
  print_process_blocks_info(blocks, num_blocks, rank, size);

  // Выделение памяти для матриц
  float (*C)[ni][nj]; C = (float(*)[ni][nj])malloc(ni * nj * sizeof(float));
  float (*A)[ni][nk]; A = (float(*)[ni][nk])malloc(ni * nk * sizeof(float));
  float (*B)[nk][nj]; B = (float(*)[nk][nj])malloc(nk * nj * sizeof(float));

  // Инициализация массивов с использованием предварительно вычисленных блоков
  init_array_with_blocks(ni, nj, nk, &alpha, &beta, *C, *A, *B, 
                         blocks, num_blocks, rank);

  // Рассылка A и B всем процессам
  MPI_Bcast(*A, ni * nk, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(*B, nk * nj, MPI_FLOAT, 0, MPI_COMM_WORLD);

  bench_timer_start();

  // Вычисление GEMM
  kernel_gemm_with_blocks(ni, nj, nk, alpha, beta, *C, *A, *B, 
                          blocks, num_blocks);

  // Сбор результатов (остается без изменений)
  if (rank == 0) {
      MPI_Reduce(MPI_IN_PLACE, *C, ni * nj, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
      MPI_Reduce(*C, NULL, ni * nj, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  }

  bench_timer_stop();

  if (rank == 0) {
      printf("\nTime: ");
      bench_timer_print();

      //if (argc > 42 && ! strcmp(argv[0], ""))
      printf("\nMATRIX C AFTER CALCULATION:\n");
      print_array(ni, nj, *C);
  }

  // Освобождение памяти
  if (blocks != NULL) {
      free(blocks);
  }
  free((void*)C);
  free((void*)A);
  free((void*)B);

  MPI_Finalize();
  return 0;
}
