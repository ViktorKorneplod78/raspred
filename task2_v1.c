// mpi_gemm_ulfm_fixed.c
/* сборка + запуск 
mpicc -O2 mpi_gemm_ulfm_fixed.c -o mpi_gemm_ulfm_fixed
mpirun --with-ft ulfm -np 6 --oversubscribe ./mpi_gemm_ulfm_fixed */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/time.h>
#include <mpi.h>
#include <mpi-ext.h>

#define NI 4
#define NJ 4
#define NK 4
#define BLOCK_SIZE 2

#define FAIL_WORLD_RANK 2
#define FAIL_KK 2

static double bench_t_start, bench_t_end;

// Структура для хранения информации о блоке
typedef struct {
    int start_i;
    int start_j;
    int end_i;
    int end_j;
    int block_id;
} BlockInfo;

// ULFM переменные
static MPI_Comm main_comm;
static int process_count, process_rank;
static int world_size0, world_rank0;
static int spare_count = 2;
static int desired_active;

// Информация о текущем состоянии вычислений
static int kk_current = 0;
static int checkpoint_interval = 1;

// Матрицы
static float *A = NULL;
static float *B = NULL;
static float *C = NULL;

// Информация о блоках текущего процесса
static BlockInfo *blocks = NULL;
static int num_blocks = 0;

static double rtclock() {
    struct timeval Tp;
    gettimeofday(&Tp, NULL);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start() { bench_t_start = rtclock(); }
void bench_timer_stop() { bench_t_end = rtclock(); }
void bench_timer_print() { printf("%0.6lf\n", bench_t_end - bench_t_start); }

// Логирование
#define ROOTLOG(fmt, ...) do { \
    if (process_rank == 0) { \
        printf("[root] " fmt "\n", ##__VA_ARGS__); \
        fflush(stdout); \
    } \
} while (0)

#define RANKLOG(fmt, ...) do { \
    printf("[rank %d] " fmt "\n", process_rank, ##__VA_ARGS__); \
    fflush(stdout); \
} while (0)

// Функция для определения принадлежности блока процессу
static int block_belongs_to_process(int block_i, int block_j, 
                                    int num_blocks_j, int rank, int size) {
    int block_id = block_i * num_blocks_j + block_j;
    return (block_id % size == rank);
}

// Получение списка блоков для процесса
static BlockInfo* get_process_blocks(int ni, int nj, int rank, int size, int *num_blocks) {
    int num_blocks_i = (ni + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_j = (nj + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
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
    
    BlockInfo *blocks = (BlockInfo*)malloc(count * sizeof(BlockInfo));
    int idx = 0;
    
    for (int block_i = 0; block_i < num_blocks_i; block_i++) {
        for (int block_j = 0; block_j < num_blocks_j; block_j++) {
            if (block_belongs_to_process(block_i, block_j, num_blocks_j, rank, size)) {
                blocks[idx].start_i = block_i * BLOCK_SIZE;
                blocks[idx].start_j = block_j * BLOCK_SIZE;
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

// Освобождение ресурсов
static void free_resources(void) {
    free(blocks);
    blocks = NULL;
    num_blocks = 0;
    
    free(A); A = NULL;
    free(B); B = NULL;
    free(C); C = NULL;
}

// Выделение памяти для матриц
static int allocate_matrices(void) {
    A = (float*)malloc(NI * NK * sizeof(float));
    B = (float*)malloc(NK * NJ * sizeof(float));
    C = (float*)malloc(NI * NJ * sizeof(float));
    
    if (!A || !B || !C) {
        free_resources();
        return -1;
    }
    
    return 0;
}

// Инициализация матриц
static void init_matrices(float alpha, float beta) {
    if (process_rank == 0) {
        for (int i = 0; i < NI; i++) {
            for (int j = 0; j < NK; j++) {
                A[i * NK + j] = (float)(i * (j + 1) % NK) / NK;
            }
        }
        for (int i = 0; i < NK; i++) {
            for (int j = 0; j < NJ; j++) {
                B[i * NJ + j] = (float)(i * (j + 2) % NJ) / NJ;
            }
        }
    }
    
    memset(C, 0, NI * NJ * sizeof(float));
    for (int b = 0; b < num_blocks; b++) {
        for (int i = blocks[b].start_i; i < blocks[b].end_i; i++) {
            for (int j = blocks[b].start_j; j < blocks[b].end_j; j++) {
                C[i * NJ + j] = (float)((i * j + 1) % NI) / NI;
            }
        }
    }
    
    for (int b = 0; b < num_blocks; b++) {
        for (int i = blocks[b].start_i; i < blocks[b].end_i; i++) {
            for (int j = blocks[b].start_j; j < blocks[b].end_j; j++) {
                C[i * NJ + j] *= beta;
            }
        }
    }
}

// Запись контрольной точки
static int checkpoint_write(int kk) {
    MPI_File fh;
    MPI_Status st;
    
    int err = MPI_File_open(main_comm, "gemm_checkpoint.bin",
                           MPI_MODE_WRONLY | MPI_MODE_CREATE,
                           MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS) {
        if (process_rank == 0) ROOTLOG("ERROR: MPI_File_open(write) err=%d", err);
        return err;
    }
    
    if (process_rank == 0) {
        err = MPI_File_write_at(fh, 0, &kk, 1, MPI_INT, &st);
        if (err != MPI_SUCCESS) {
            MPI_File_close(&fh);
            return err;
        }
    }
    
    MPI_Barrier(main_comm);
    
    for (int b = 0; b < num_blocks; b++) {
        for (int i = blocks[b].start_i; i < blocks[b].end_i; i++) {
            int start_j = blocks[b].start_j;
            int end_j = blocks[b].end_j;
            int count = end_j - start_j;
            
            MPI_Offset offset = sizeof(int) + 
                              ((size_t)i * NJ + start_j) * sizeof(float);
            
            err = MPI_File_write_at(fh, offset, 
                                   &C[i * NJ + start_j], 
                                   count, MPI_FLOAT, &st);
            if (err != MPI_SUCCESS) {
                MPI_File_close(&fh);
                return err;
            }
        }
    }
    
    MPI_File_close(&fh);
    
    if (process_rank == 0 && (kk % (BLOCK_SIZE * 10) == 0 || kk == NK)) {
        ROOTLOG("Checkpoint written at kk=%d", kk);
    }
    
    return MPI_SUCCESS;
}

// Чтение контрольной точки
static int checkpoint_read(int *kk) {
    MPI_File fh;
    
    int err = MPI_File_open(main_comm, "gemm_checkpoint.bin",
                           MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS) {
        if (process_rank == 0) ROOTLOG("No checkpoint found");
        return 0;
    }
    
    if (process_rank == 0) {
        err = MPI_File_read_at(fh, 0, kk, 1, MPI_INT, MPI_STATUS_IGNORE);
        if (err != MPI_SUCCESS) {
            MPI_File_close(&fh);
            return -1;
        }
    }
    
    MPI_Bcast(kk, 1, MPI_INT, 0, main_comm);
    
    if (*kk < 0 || *kk > NK) {
        if (process_rank == 0) ROOTLOG("Invalid checkpoint kk=%d", *kk);
        MPI_File_close(&fh);
        return 0;
    }
    
    for (int b = 0; b < num_blocks; b++) {
        for (int i = blocks[b].start_i; i < blocks[b].end_i; i++) {
            int start_j = blocks[b].start_j;
            int end_j = blocks[b].end_j;
            int count = end_j - start_j;
            
            MPI_Offset offset = sizeof(int) + 
                              ((size_t)i * NJ + start_j) * sizeof(float);
            
            err = MPI_File_read_at(fh, offset, 
                                  &C[i * NJ + start_j], 
                                  count, MPI_FLOAT, MPI_STATUS_IGNORE);
            if (err != MPI_SUCCESS) {
                MPI_File_close(&fh);
                return -1;
            }
        }
    }
    
    MPI_File_close(&fh);
    
    if (process_rank == 0) {
        ROOTLOG("Checkpoint loaded, kk=%d", *kk);
    }
    
    return 1;
}

// Вычисление одного блока по k
static int compute_k_block(int kk_start, float alpha) {
    int kk_end = kk_start + BLOCK_SIZE;
    if (kk_end > NK) kk_end = NK;
    
    for (int b = 0; b < num_blocks; b++) {
        for (int i = blocks[b].start_i; i < blocks[b].end_i; i++) {
            for (int j = blocks[b].start_j; j < blocks[b].end_j; j++) {
                for (int k = kk_start; k < kk_end; k++) {
                    C[i * NJ + j] += alpha * A[i * NK + k] * B[k * NJ + j];
                }
            }
        }
    }
    
    return MPI_SUCCESS;
}

// Основная функция вычислений
static int run_gemm(float alpha, float beta) {
    static int failure_simulated = 0;
    
    // Рассылка A и B всем процессам
    int err = MPI_Bcast(A, NI * NK, MPI_FLOAT, 0, main_comm);
    if (err != MPI_SUCCESS) return err;
    
    err = MPI_Bcast(B, NK * NJ, MPI_FLOAT, 0, main_comm);
    if (err != MPI_SUCCESS) return err;
    
    for (int kk = kk_current; kk < NK; kk += BLOCK_SIZE) {
        // Симуляция сбоя
        if (!failure_simulated && world_rank0 == FAIL_WORLD_RANK && kk == FAIL_KK * BLOCK_SIZE) {
            RANKLOG("Simulating failure at kk=%d (world_rank0=%d)", kk, world_rank0);
            failure_simulated = 1;
            raise(SIGKILL);
        }
        
        err = compute_k_block(kk, alpha);
        if (err != MPI_SUCCESS) return err;
        
        if ((kk / BLOCK_SIZE) % checkpoint_interval == 0 || kk + BLOCK_SIZE >= NK) {
            err = checkpoint_write(kk + BLOCK_SIZE);
            if (err != MPI_SUCCESS) return err;
        }
    }
    
    return MPI_SUCCESS;
}

// Восстановление после сбоя
static int recovery_procedure(float alpha, float beta) {
    ROOTLOG("Failure detected, revoke+shrink communicator");
    
    MPIX_Comm_revoke(main_comm);
    
    MPI_Comm new_comm;
    MPIX_Comm_shrink(main_comm, &new_comm);
    MPI_Comm_free(&main_comm);
    main_comm = new_comm;
    
    MPI_Comm_set_errhandler(main_comm, MPI_ERRORS_RETURN);
    MPI_Comm_size(main_comm, &process_count);
    MPI_Comm_rank(main_comm, &process_rank);
    
    ROOTLOG("New communicator: size=%d", process_count);
    
    // Обновляем desired_active
    if (desired_active > process_count) {
        desired_active = process_count;
    }
    
    // Перераспределяем блоки
    free_resources();
    if (allocate_matrices() != 0) {
        ROOTLOG("ERROR: Allocation failed after shrink");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // ВАЖНО: ВСЕ процессы получают блоки после shrink
    blocks = get_process_blocks(NI, NJ, process_rank, process_count, &num_blocks);
    
    // Восстановление из контрольной точки
    int kk_from_cp = 0;
    int cp_res = checkpoint_read(&kk_from_cp);
    
    if (cp_res <= 0) {
        kk_current = 0;
        init_matrices(alpha, beta);
        ROOTLOG("No checkpoint found, starting from scratch");
    } else {
        kk_current = kk_from_cp;
        MPI_Bcast(A, NI * NK, MPI_FLOAT, 0, main_comm);
        MPI_Bcast(B, NK * NJ, MPI_FLOAT, 0, main_comm);
        ROOTLOG("Restarting from checkpoint kk=%d", kk_current);
    }
    
    return MPI_SUCCESS;
}

// Проверка результата
static int verify_result(void) {
    double sum_local = 0.0, sum_global;
    
    for (int b = 0; b < num_blocks; b++) {
        for (int i = blocks[b].start_i; i < blocks[b].end_i; i++) {
            for (int j = blocks[b].start_j; j < blocks[b].end_j; j++) {
                sum_local += C[i * NJ + j];
            }
        }
    }
    
    int err = MPI_Reduce(&sum_local, &sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, main_comm);
    if (err != MPI_SUCCESS) return err;
    
    if (process_rank == 0) {
        ROOTLOG("Verification sum = %f", sum_global);
    }
    
    return MPI_SUCCESS;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    main_comm = MPI_COMM_WORLD;
    MPI_Comm_set_errhandler(main_comm, MPI_ERRORS_RETURN);
    
    MPI_Comm_size(main_comm, &process_count);
    MPI_Comm_rank(main_comm, &process_rank);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank0);
    world_size0 = process_count;
    
    if (spare_count >= world_size0) spare_count = 0;
    desired_active = world_size0 - spare_count;
    if (desired_active < 1) desired_active = world_size0;
    
    if (process_rank == 0) {
        ROOTLOG("GEMM start: NI=%d, NJ=%d, NK=%d", NI, NJ, NK);
        ROOTLOG("Total ranks=%d, active=%d, spare=%d", 
                world_size0, desired_active, spare_count);
    }
    
    // Инициализация блоков для активных процессов
    if (process_rank < desired_active) {
        blocks = get_process_blocks(NI, NJ, process_rank, desired_active, &num_blocks);
    } else {
        // Резервные процессы не имеют блоков
        blocks = NULL;
        num_blocks = 0;
    }
    
    if (allocate_matrices() != 0) {
        if (process_rank == 0) ROOTLOG("ERROR: Allocation failed");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    float alpha = 1.5, beta = 1.2;
    
    int cp_res = checkpoint_read(&kk_current);
    if (cp_res <= 0) {
        kk_current = 0;
        init_matrices(alpha, beta);
    }
    
    bench_timer_start();
    
    // Основной цикл с восстановлением
    while (1) {
        int err = run_gemm(alpha, beta);
        if (err == MPI_SUCCESS) break;
        
        err = recovery_procedure(alpha, beta);
        if (err != MPI_SUCCESS) {
            ROOTLOG("FATAL: Recovery failed");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }
    
    bench_timer_stop();
    
    // ВАЖНОЕ ИСПРАВЛЕНИЕ: Все процессы участвуют в MPI_Reduce
    if (process_rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, C, NI * NJ, MPI_FLOAT, MPI_SUM, 0, main_comm);
    } else {
        // Все остальные процессы (включая бывшие резервные) участвуют
        MPI_Reduce(C, NULL, NI * NJ, MPI_FLOAT, MPI_SUM, 0, main_comm);
    }
    
    verify_result();
    
    if (process_rank == 0) {
        printf("\nTime: ");
        bench_timer_print();
        
        printf("\nFirst 4x4 of matrix C:\n");
        for (int i = 0; i < 4 && i < NI; i++) {
            for (int j = 0; j < 4 && j < NJ; j++) {
                printf("%0.2f ", C[i * NJ + j]);
            }
            printf("\n");
        }
    }
    
    free_resources();
    
    if (process_rank == 0) ROOTLOG("Finalizing MPI");
    MPI_Finalize();
    return 0;
}
