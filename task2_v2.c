#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/time.h>
#include <mpi.h>
#include <mpi-ext.h>

#define NI 8
#define NJ 8
#define NK 8
#define BLOCK_SIZE 2

#define FAIL_RANK 2
#define FAIL_K 0
#define CHECKPOINT_FILENAME "checkpoint.bin"

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
static int k_current = 0;
static int checkpoint_interval = 1;

// Матрицы
static float *A = NULL;
static float *B = NULL;
static float *C = NULL;

// Информация о блоках текущего процесса
static BlockInfo *blocks = NULL;
static int num_blocks = 0;

// Для расчёта времени работы GEMM
static double rtclock() {
    struct timeval Tp;
    gettimeofday(&Tp, NULL);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}
void bench_timer_start() { bench_t_start = rtclock(); }
void bench_timer_stop() { bench_t_end = rtclock(); }
void bench_timer_print() { printf("%0.6lf\n", bench_t_end - bench_t_start); }

// Логирование
#define ROOTLOG(fmt, ...) do {\
    if (process_rank == 0) {\
        printf("[root] " fmt "\n", ##__VA_ARGS__);\
        fflush(stdout);\
    }\
} while (0)

#define RANKLOG(fmt, ...) do {\
    printf("[rank %d] " fmt "\n", process_rank, ##__VA_ARGS__);\
    fflush(stdout);\
} while (0)

// Определение принадлежности блока процессу
static int block_belongs_to_process(int block_i, int block_j, int num_blocks_j, int rank, int size) {
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
}

// Запись контрольной точки
static int checkpoint_write(int k) {
    MPI_File fh;
    MPI_Status st;
    
    int err = MPI_File_open(main_comm, CHECKPOINT_FILENAME, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS) {
        ROOTLOG("ERROR: MPI_File_open(write) err=%d", err);
        return err;
    }
    
    if (process_rank == 0) {
        err = MPI_File_write_at(fh, 0, &k, 1, MPI_INT, &st);
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
            
            MPI_Offset offset = sizeof(int) + ((size_t)i * NJ + start_j) * sizeof(float);
            
            err = MPI_File_write_at(fh, offset, &C[i * NJ + start_j], count, MPI_FLOAT, &st);
            if (err != MPI_SUCCESS) {
                MPI_File_close(&fh);
                return err;
            }
        }
    }
    
    MPI_File_close(&fh);
    //RANKLOG("Checkpoint written at k=%d", k);
    
    return MPI_SUCCESS;
}

// Чтение контрольной точки
static int checkpoint_read(int *k) {
    MPI_File fh;
    
    int err = MPI_File_open(main_comm, CHECKPOINT_FILENAME, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS) {
        ROOTLOG("No checkpoint found");
        return 0;
    }
    
    if (process_rank == 0) {
        err = MPI_File_read_at(fh, 0, k, 1, MPI_INT, MPI_STATUS_IGNORE);
        if (err != MPI_SUCCESS) {
            MPI_File_close(&fh);
            return -1;
        }
    }
    
    MPI_Bcast(k, 1, MPI_INT, 0, main_comm);
    
    if (*k < 0 || *k > NK) {
        ROOTLOG("Invalid checkpoint k=%d", *k);
        MPI_File_close(&fh);
        return 0;
    }
    
    for (int b = 0; b < num_blocks; b++) {
        for (int i = blocks[b].start_i; i < blocks[b].end_i; i++) {
            int start_j = blocks[b].start_j;
            int end_j = blocks[b].end_j;
            int count = end_j - start_j;
            
            MPI_Offset offset = sizeof(int) + ((size_t)i * NJ + start_j) * sizeof(float);
            
            err = MPI_File_read_at(fh, offset, &C[i * NJ + start_j], count, MPI_FLOAT, MPI_STATUS_IGNORE);
            if (err != MPI_SUCCESS) {
                MPI_File_close(&fh);
                return -1;
            }
        }
    }
    
    MPI_File_close(&fh);
    
    RANKLOG("Checkpoint loaded, k=%d", *k);
    
    return 1;
}

// Вычисление одного блока по k
static int compute_k_block(int k_start, float alpha, float beta) {
    int k_end = k_start + BLOCK_SIZE;
    if (k_end > NK) k_end = NK;
    
    for (int b = 0; b < num_blocks; b++) {
        for (int i = blocks[b].start_i; i < blocks[b].end_i; i++) {
            for (int j = blocks[b].start_j; j < blocks[b].end_j; j++) {
                if (k_start == 0) {
                    C[i * NJ + j] *= beta;
                }
                for (int k = k_start; k < k_end; k++) {
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
    
    for (int k = k_current; k < NK; k += BLOCK_SIZE) {
        // Симуляция сбоя
        if (!failure_simulated && world_rank0 == FAIL_RANK && k == FAIL_K) {
            RANKLOG("Simulating failure at k=%d (world_rank0=%d)", k, world_rank0);
            failure_simulated = 1;
            raise(SIGKILL);
        }
        
        err = compute_k_block(k, alpha, beta);
        if (err != MPI_SUCCESS) return err;
        
        if ((k / BLOCK_SIZE) % checkpoint_interval == 0 || k + BLOCK_SIZE >= NK) {
            err = checkpoint_write(k + BLOCK_SIZE);
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

    // Рассчитываем новое кол-во резервных и активных процессов
    if (spare_count > 0) {
        spare_count--;
    }
    // Активных должно остаться столько же, сколько было до сбоя
    int new_active = process_count - spare_count;

    ROOTLOG("New communicator: size=%d: %d active, %d spare", process_count, new_active, spare_count);
    if (new_active < 1) {
        ROOTLOG("ERROR: Not enough processes left");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Перераспределяем блоки на активные процессы
    free_resources();
    if (allocate_matrices() != 0) {
        ROOTLOG("ERROR: Allocation failed after shrink");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (process_rank < new_active) {
        blocks = get_process_blocks(NI, NJ, process_rank, new_active, &num_blocks);
        RANKLOG("Active process with %d blocks", num_blocks);
    } else {
        // Spare процессы остаются без блоков
        blocks = NULL;
        num_blocks = 0;
        RANKLOG("Spare process - no blocks assigned");
    }

    // Восстановление из контрольной точки
    int k_from_cp = 0;
    int cp_res = checkpoint_read(&k_from_cp);

    if (cp_res <= 0) {
        k_current = 0;
        // Инициализируем матрицы (только активные процессы инициализируют свои блоки C)
        init_matrices(alpha, beta);
        ROOTLOG("No checkpoint found, starting from scratch");
    } else {
        k_current = k_from_cp;
        // Рассылаем A и B всем процессам (и активным, и spare)
        MPI_Bcast(A, NI * NK, MPI_FLOAT, 0, main_comm);
        MPI_Bcast(B, NK * NJ, MPI_FLOAT, 0, main_comm);
        ROOTLOG("Restarting from checkpoint kk=%d", k_current);
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

    // Инициализация spare_count
    spare_count = 2; // Например, 2 spare процесса
    if (spare_count >= world_size0) spare_count = 0;

    desired_active = world_size0 - spare_count;
    if (desired_active < 1) desired_active = world_size0;

    ROOTLOG("GEMM start: NI=%d, NJ=%d, NK=%d", NI, NJ, NK);
    ROOTLOG("Total ranks=%d, active=%d, spare=%d", world_size0, desired_active, spare_count);

    // Инициализация блоков для активных процессов
    if (process_rank < desired_active) {
        blocks = get_process_blocks(NI, NJ, process_rank, desired_active, &num_blocks);
    } else {
        // Spare процессы не имеют блоков
        blocks = NULL;
        num_blocks = 0;
    }

    if (allocate_matrices() != 0) {
        ROOTLOG("ERROR: Allocation failed");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    float alpha = 1.5, beta = 1.2;

    int cp_res = checkpoint_read(&k_current);
    if (cp_res <= 0) {
        k_current = 0;
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

    if (process_rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, C, NI * NJ, MPI_FLOAT, MPI_SUM, 0, main_comm);
    } else {
        MPI_Reduce(C, NULL, NI * NJ, MPI_FLOAT, MPI_SUM, 0, main_comm);
    }

    if (process_rank == 0) {
        printf("\nTime: ");
        bench_timer_print();

        printf("\nResult on matrix C:\n");
        for (int i = 0; i < NI; i++) {
            for (int j = 0; j < NJ; j++) {
                printf("%8.4f ", C[i * NJ + j]);
            }
            printf("\n");
        }

        //print_array(NI, NJ, C);
    }

    free_resources();

    ROOTLOG("Finalizing MPI");
    MPI_Finalize();
    return 0;
}
