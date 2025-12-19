#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 5
#define BLOCK_SIZE 4  // 4 int = 16 байт

int get_rank(int row, int col) {
    return row * N + col;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != N * N) {
        if (rank == 0) printf("Need exactly %d processes\n", N * N);
        MPI_Finalize();
        return 1;
    }

    int row = rank / N;
    int col = rank % N;

    // Буфер размера, достаточного для хранения данных от всех процессов
    int *data = (int*)malloc(N * N * BLOCK_SIZE * sizeof(int));
    // Длина хранящихся на процессе данных (изначально у всех по 4 инта)
    int data_count = BLOCK_SIZE;

    // Инициализация данных рангом процесса
    for (int i = 0; i < BLOCK_SIZE; i++) {
        data[i] = rank;
    }

    MPI_Status status;
    int dest, src;

    // ================= ШАГ 1 =================
    if (col == 4) {
        dest = get_rank(row, 3);
        MPI_Ssend(data, BLOCK_SIZE, MPI_INT, dest, 1, MPI_COMM_WORLD);
    }

    if (col == 3) {
        src = get_rank(row, 4);
        MPI_Recv(data + data_count, BLOCK_SIZE, MPI_INT, src, 1,
                 MPI_COMM_WORLD, &status);
        data_count += BLOCK_SIZE;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ================= ШАГ 2 =================
    if (row == 4 && col < 4) {
        dest = get_rank(3, col);
        int send_count = (col == 3) ? 8 : 4;
        MPI_Ssend(data, send_count, MPI_INT, dest, 2, MPI_COMM_WORLD);
    }

    if (row == 3 && col < 4) {
        src = get_rank(4, col);
        int recv_count = (col == 3) ? 8 : 4;
        MPI_Recv(data + data_count, recv_count, MPI_INT, src, 2,
                 MPI_COMM_WORLD, &status);
        data_count += recv_count;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ================= ШАГ 3 =================
    if (row < 4) {
        if (col == 1) {
            dest = get_rank(row, 0);
            int send_count = (row == 3) ? 8 : 4;
            MPI_Ssend(data, send_count, MPI_INT, dest, 3, MPI_COMM_WORLD);
        }
        else if (col == 3) {
            dest = get_rank(row, 2);
            int send_count = (row == 3) ? 16 : 8;
            MPI_Ssend(data, send_count, MPI_INT, dest, 3, MPI_COMM_WORLD);
        }
        else if (col == 0) {
            src = get_rank(row, 1);
            int recv_count = (row == 3) ? 8 : 4;
            MPI_Recv(data + data_count, recv_count, MPI_INT, src, 3,
                     MPI_COMM_WORLD, &status);
            data_count += recv_count;
        }
        else if (col == 2) {
            src = get_rank(row, 3);
            int recv_count = (row == 3) ? 16 : 8;
            MPI_Recv(data + data_count, recv_count, MPI_INT, src, 3,
                     MPI_COMM_WORLD, &status);
            data_count += recv_count;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ================= ШАГ 4 ================
    if (row == 1 && col == 0) {
        dest = get_rank(0, 0);
        MPI_Ssend(data, 8, MPI_INT, dest, 4, MPI_COMM_WORLD);
    }
    else if (row == 3 && col == 0) {
        dest = get_rank(2, 0);
        MPI_Ssend(data, 16, MPI_INT, dest, 4, MPI_COMM_WORLD);
    }
    else if (row == 1 && col == 2) {
        dest = get_rank(0, 2);
        MPI_Ssend(data, 12, MPI_INT, dest, 4, MPI_COMM_WORLD);
    }
    else if (row == 3 && col == 2) {
        dest = get_rank(2, 2);
        MPI_Ssend(data, 24, MPI_INT, dest, 4, MPI_COMM_WORLD);
    }
    else if (row == 0 && col == 0) {
        src = get_rank(1, 0);
        MPI_Recv(data + data_count, 8, MPI_INT, src, 4, MPI_COMM_WORLD, &status);
        data_count += 8;
    }
    else if (row == 2 && col == 0) {
        src = get_rank(3, 0);
        MPI_Recv(data + data_count, 16, MPI_INT, src, 4, MPI_COMM_WORLD, &status);
        data_count += 16;
    }
    else if (row == 0 && col == 2) {
        src = get_rank(1, 2);
        MPI_Recv(data + data_count, 12, MPI_INT, src, 4, MPI_COMM_WORLD, &status);
        data_count += 12;
    }
    else if (row == 2 && col == 2) {
        src = get_rank(3, 2);
        MPI_Recv(data + data_count, 24, MPI_INT, src, 4, MPI_COMM_WORLD, &status);
        data_count += 24;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ================= ШАГ 5 =================
    if (row == 0 && col == 2) {
        dest = get_rank(0, 0);
        MPI_Ssend(data, 24, MPI_INT, dest, 5, MPI_COMM_WORLD);
    }
    else if (row == 2 && col == 2) {
        dest = get_rank(2, 0);
        MPI_Ssend(data, 36, MPI_INT, dest, 5, MPI_COMM_WORLD);
    }
    else if (row == 0 && col == 0) {
        src = get_rank(0, 2);
        MPI_Recv(data + data_count, 24, MPI_INT, src, 5, MPI_COMM_WORLD, &status);
        data_count += 24;
    }
    else if (row == 2 && col == 0) {
        src = get_rank(2, 2);
        MPI_Recv(data + data_count, 36, MPI_INT, src, 5, MPI_COMM_WORLD, &status);
        data_count += 36;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ================= ШАГ 6 =================
    if (row == 2 && col == 0) {
        dest = get_rank(0, 0);
        MPI_Ssend(data, 60, MPI_INT, dest, 6, MPI_COMM_WORLD);
    }
    else if (rank == 0) {
        src = get_rank(2, 0);
        MPI_Recv(data + data_count, 60, MPI_INT, src, 6, MPI_COMM_WORLD, &status);
        data_count += 60;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Проверка данных
    if (rank == 0) {
        int count[25] = {0};
        for (int i = 0; i < data_count; i++) {
            count[data[i]]++;
        }
        int correct = 1;
        for (int i = 0; i < 25; i++) {
            if (count[i] != BLOCK_SIZE) {
                printf("Error: process %d appears %d times\n", i, count[i]);
                correct = 0;
            }
        }
        if (correct) {
            printf("SUCCESS: All data gathered correctly!\n");
        }
    }

    free(data);
    MPI_Finalize();
    return 0;
}
