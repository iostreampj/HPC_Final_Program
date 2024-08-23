/*
  author: Jiawei Zhang
  DESCRIPTION:Parallel Gaussian elimination method for solving systems of linear equations
  Date of creation：2024-7-11
*/

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <iostream>

#define LOOP_UNROLL_FACTOR 4
const int b = 8;  // Block size, must divide N

static int mpiRankID = -1;
static int mpiProcessNum = -1;

// Function prototypes
void print_matrix(double** T, int rows, int cols);
void Sequential_computation(double** A, int n);
typedef int (*tfMPI_TxRx)(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm);

class CMpiCal {
public:
    CMpiCal() {}

    // Initialize the MPI environment and necessary variables
    int Init(double** A, int argc, char** argv) {
        int rst;
        if (argc != 2) {
            printf("Usage: mpirun -np [mpiProcessNum] ./program2 [matrix size N]\n");
            exit(-1);
            return -1;
        }
        memset(this, 0, sizeof(*this));
        N = atoi(argv[1]);

        if (N % b) {
            printf("block size b must be divisible by matrix size N\n");
            exit(-1);
        }
        this->A = A;
        if (N < 1) {
            perror("matrix size N format error\n");
            exit(-2);
        }
        if (b < 1) {
            perror("block size b format error\n");
            exit(-2);
        }
        rst = MPI_Init(&argc, &argv);
        if (rst != 0)
            return rst;
        rst = MPI_Comm_size(MPI_COMM_WORLD, &mpiProcessNum);
        if (rst != 0)
            return rst;
        rst = MPI_Comm_rank(MPI_COMM_WORLD, &mpiRankID);
        if (rst != 0)
            return rst;
        rst = initGlobalVars();
        if (rst != 0)
            return rst;
        rst = initTmp1();
        if (rst != 0)
            return rst;
        rst = MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes
        if (rst != 0)
            return rst;

        return 0;
    }

    // Clean up and finalize the MPI environment
    int DInit() {
        arr_dbl_2d_free(priC_b_b);
        arr_dbl_2d_free(shmA_N_Y);
        arr_dbl_2d_free(shmB_N_b);
        free(k_prev_list);
        free(indk_next_list);
        MPI_Type_free(&type_vec_N_b_N);
        MPI_Type_free(&type_vec_N_b_Y);
        MPI_Type_free(&type_bcast_N_b_2d_calBlockCnt);
        MPI_Finalize();
        return 0;
    }

protected:
    // Custom MPI receive function with status checking
    static inline int MPI_RecvEx(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm) {
        static MPI_Status status;
        return MPI_Recv(buf, count, datatype, source, tag, comm, &status);
    }

    // Print formatted messages with MPI rank tag
    void mpi_printf(const char* tag, const char* pformat, ...) {
        va_list args;
        va_start(args, pformat);
        printf("[%d][%s]", mpiRankID, tag);
        vprintf(pformat, args);
        va_end(args);
    }

    // Initialize the block memory used in the computation
    int initTmp1() {
        return MatrixMergeOrSplit(true);
    }

    // Allocate and initialize a 2D array
    static double** arr_dbl_2d_create(int rowNum, int colNum, double** pp_ar_1d = nullptr) {
        double* ar_1d = (double*)calloc(rowNum * colNum, sizeof(double));
        double** ar_2d = (double**)calloc(rowNum, sizeof(double*));
        if (pp_ar_1d != nullptr) {
            *pp_ar_1d = ar_1d;
        }
        int i;
        for (i = 0; i < rowNum; i++) {
            ar_2d[i] = ar_1d + i * colNum;
        }
        return ar_2d;
    }

    // Free the allocated 2D array
    void arr_dbl_2d_free(double** ar_2d) {
        free(ar_2d[0]);
        free(ar_2d);
    }

    // Initialize global MPI data types and matrix variables
    int initGlobalVars() {
        MPI_Type_vector(N, b, N, MPI_DOUBLE, &type_vec_N_b_N);
        MPI_Type_commit(&type_vec_N_b_N);
        columnBlockNum = N / b;
        Y = columnBlockNum / mpiProcessNum * b;
        if (mpiRankID < columnBlockNum % mpiProcessNum) {
            Y += b;
        }
        shmA_N_Y = arr_dbl_2d_create(N, Y);
        priC_b_b = arr_dbl_2d_create(b, b, &priC_b_b_1d);
        shmB_N_b = arr_dbl_2d_create(N, b);
        k_prev_list = (int*)calloc(b, sizeof(int));
        indk_next_list = (int*)calloc(b, sizeof(int));

        assert(indk_next_list != nullptr);
        MPI_Type_vector(N, b, Y, MPI_DOUBLE, &type_vec_N_b_Y);
        MPI_Type_commit(&type_vec_N_b_Y);
        return 0;
    }

    // Find pivot elements in local blocks and record the operations
    int findAndRecordLocal() {
        int rst, i, j, k, l, indk;
        double c, amax;
        for (i = columnStart, k = rowStart; i < columnEnd; i++, k++) {
            k_prev_list[i - columnStart] = k;
            amax = shmA_N_Y[k][i];
            indk = k;
            for (j = k + 1; j < N; j++) {
                if (fabs(shmA_N_Y[j][i]) > fabs(amax)) {
                    amax = shmA_N_Y[j][i];
                    indk = j;
                }
            }
            indk_next_list[i - columnStart] = indk;
            if (amax == 0) {
                printf("Matrix is singular!\n");
                exit(1);
            }
            if (indk != k) {
                for (j = 0; j < Y; ++j) {
                    c = shmA_N_Y[k][j];
                    shmA_N_Y[k][j] = shmA_N_Y[indk][j];
                    shmA_N_Y[indk][j] = c;
                }
            }
            for (j = k + 1; j < N; j++) {
                shmA_N_Y[j][i] /= shmA_N_Y[k][i];
            }
            for (j = k + 1; j < N; j++) {
                c = shmA_N_Y[j][i];
                for (l = i + 1; l < columnEnd; l++) {
                    shmA_N_Y[j][l] -= c * shmA_N_Y[k][l];
                }
            }
        }
        rst = MPI_Bcast(k_prev_list, b, MPI_INT, iProcessColumn, MPI_COMM_WORLD);
        if (rst != 0)
            return rst;
        rst = MPI_Bcast(indk_next_list, b, MPI_INT, iProcessColumn, MPI_COMM_WORLD);
        if (rst != 0)
            return rst;
        for (i = calBlockCnt; i < calBlockCnt + b; i++) {
            memcpy(shmB_N_b[i], &shmA_N_Y[i][columnStart], (i - calBlockCnt) * sizeof(double));
        }
        return 0;
    }

    // Sync the pivot elements across all processes
    int findAndRecordSync() {
        int rst, i, j;
        double c;
        rst = MPI_Bcast(k_prev_list, b, MPI_INT, iProcessColumn, MPI_COMM_WORLD);
        if (rst != 0)
            return rst;
        rst = MPI_Bcast(indk_next_list, b, MPI_INT, iProcessColumn, MPI_COMM_WORLD);
        if (rst != 0)
            return rst;
        for (i = 0; i < b; i++) {
            auto& shmA_N_Y_prev = shmA_N_Y[k_prev_list[i]];
            auto& shmA_N_Y_next = shmA_N_Y[indk_next_list[i]];
            if (k_prev_list[i] != indk_next_list[i]) {
                for (j = 0; j < Y; ++j) {
                    c = shmA_N_Y_prev[j];
                    shmA_N_Y_prev[j] = shmA_N_Y_next[j];
                    shmA_N_Y_next[j] = c;
                }
            }
        }
        return 0;
    }

    // Apply the Gaussian elimination step to reduce the matrix
    int ShrinkageMatrix() {
        int rst;
        int i, j, k, l, m ,n;
        rst = MPI_Bcast(shmB_N_b[calBlockCnt], 1, type_bcast_N_b_2d_calBlockCnt, iProcessColumn, MPI_COMM_WORLD);
        if (rst != 0)
            return rst;
        calBlockCnt += b;

        memset(priC_b_b[0], 0, sizeof(double) * b * b);
        double* pRowNext = priC_b_b[0];
        for (i = 0; i < b; i++) {
            const double* pRowPrev = shmB_N_b[rowStart + i];
            for (j = 0; j < b; j++, pRowPrev++, pRowNext++) {
                if (j < i)
                    *pRowNext = *pRowPrev;
                else if (j == i)
                    *pRowNext = 1;
            }
        }
        double** pShmA = &shmA_N_Y[rowStart];

        for (i = 1; i < b; i++) {
            l = i / LOOP_UNROLL_FACTOR;
            l *= LOOP_UNROLL_FACTOR;
            m = l + i % LOOP_UNROLL_FACTOR;

            for (j = processColumnJ; j < Y; j++) {
                const double* priC = priC_b_b[i];

                for (k = 0; k < l; k += LOOP_UNROLL_FACTOR, priC += LOOP_UNROLL_FACTOR) {
                    pShmA[i][j] -= priC[0] * pShmA[k][j];
                    pShmA[i][j] -= priC[1] * pShmA[k + 1][j];
                    pShmA[i][j] -= priC[2] * pShmA[k + 2][j];
                    pShmA[i][j] -= priC[3] * pShmA[k + 3][j];
                }

                for (; k < m; k++, priC++) {
                    pShmA[i][j] -= *priC * pShmA[k][j];
                }
            }
        }
        return 0;
    }

    // Dynamic processing for tail blocks in matrix
    int TailDynamicProcess(int& tailCalCnt) {
        int rst;
        int i, j, k;
        if (mpiRankID == iProcessColumn && iColumnBlock != columnBlockNum - 1) {
            for (i = tailCalCnt; i < N; i++) {
                memcpy(shmB_N_b[i], &shmA_N_Y[i][columnStart], b * sizeof(double));
            }
        }
        MPI_Datatype type_bcast_N_b_2d_tailCalCnt;
        rst = MPI_Type_vector(N - (iColumnBlock + 1) * b, b, b, MPI_DOUBLE, &type_bcast_N_b_2d_tailCalCnt);
        if (rst != 0)
            return rst;
        rst = MPI_Type_commit(&type_bcast_N_b_2d_tailCalCnt);
        if (rst != 0)
            return rst;
        rst = MPI_Bcast(shmB_N_b[tailCalCnt], 1, type_bcast_N_b_2d_tailCalCnt, iProcessColumn, MPI_COMM_WORLD);
        if (rst != 0)
            return rst;
        MPI_Type_free(&type_bcast_N_b_2d_tailCalCnt);

        double** pShmA_src = &shmA_N_Y[rowStart];
        double* pShmA_tar = shmA_N_Y[rowEnd];
        const double* pShmB = shmB_N_b[rowEnd];

        double sub_tmp;
        for (i = rowEnd; i < N; i++, pShmA_tar += Y, pShmB += b) {
            double* pShmA_tar_1 = &pShmA_tar[processColumnJ];
            for (j = processColumnJ; j < Y; j++, pShmA_tar_1++) {
                sub_tmp = *pShmA_tar_1;

                // Apply the subtraction for Gaussian elimination
                sub_tmp -= pShmB[0] * pShmA_src[0][j];
                sub_tmp -= pShmB[1] * pShmA_src[1][j];
                sub_tmp -= pShmB[2] * pShmA_src[2][j];
                sub_tmp -= pShmB[3] * pShmA_src[3][j];
                sub_tmp -= pShmB[4] * pShmA_src[4][j];
                sub_tmp -= pShmB[5] * pShmA_src[5][j];
                sub_tmp -= pShmB[6] * pShmA_src[6][j];
                sub_tmp -= pShmB[7] * pShmA_src[7][j];
                *pShmA_tar_1 = sub_tmp;
            }
        }

        tailCalCnt += b;
        return 0;
    }

    // Initialize variables for processing each column block
    void ColumnBlockVarsInit(int iColumnBlock) {
        blockID = iColumnBlock / mpiProcessNum;
        iProcessColumn = iColumnBlock % mpiProcessNum;
        columnStart = blockID * b;
        rowStart = iColumnBlock * b;
        columnEnd = columnStart + b;
        rowEnd = rowStart + b;
        processColumnJ = (mpiRankID <= iProcessColumn) ? columnEnd : columnStart;
    }

public:
    // Main computation function
    int CalRun() {
        int rst;
        int tailCalCnt = b;
        calBlockCnt = 0;
        rst = CalPreInit();
        if (rst != 0)
            return rst;
        for (iColumnBlock = 0; iColumnBlock < columnBlockNum; iColumnBlock++) {
            ColumnBlockVarsInit(iColumnBlock);
            if (mpiRankID == iProcessColumn) {
                rst = findAndRecordLocal();
                if (rst != 0)
                    return rst;
            } else {
                rst = findAndRecordSync();
                if (rst != 0)
                    return rst;
            }
            rst = ShrinkageMatrix();
            if (rst != 0)
                return rst;
            rst = TailDynamicProcess(tailCalCnt);
            if (rst != 0)
                return rst;
        }
        rst = GatherProcessRst(); // Gather results from all processes
        if (rst != 0)
            return rst;
        return 0;
    }

protected:
    // Prepare for the computation, initialize MPI data types
    int CalPreInit() {
        int array_of_blocklengths[b], array_of_displacements[b];
        int i;
        for (i = 0; i < b; i++) {
            array_of_blocklengths[i] = i;
            array_of_displacements[i] = i * b;
        }
        MPI_Type_indexed(b, array_of_blocklengths, array_of_displacements, MPI_DOUBLE, &type_bcast_N_b_2d_calBlockCnt);
        MPI_Type_commit(&type_bcast_N_b_2d_calBlockCnt);
        return 0;
    }

    // Split or merge the matrix blocks between processes
    int MatrixMergeOrSplit(bool bMerge) {
        int rst;
        int i, j, k, l, sumOther, sumLocal;
        double **srcMatrix, **tarMatrix;
        int *p_sumOther, *p_sumLocal;
        tfMPI_TxRx fMPI_Tx, fMPI_Rx;
        if (bMerge) {
            fMPI_Tx = (tfMPI_TxRx)MPI_Send;
            fMPI_Rx = MPI_RecvEx;
            srcMatrix = A;
            tarMatrix = shmA_N_Y;
            p_sumLocal = &sumLocal;
            p_sumOther = &sumOther;
        } else {
            fMPI_Tx = MPI_RecvEx;
            fMPI_Rx = (tfMPI_TxRx)MPI_Send;
            srcMatrix = shmA_N_Y;
            tarMatrix = A;
            p_sumLocal = &sumOther;
            p_sumOther = &sumLocal;
        }
        if (mpiRankID == 0) {
            sumOther = 0;
            sumLocal = 0;
            for (j = 0; j < columnBlockNum / mpiProcessNum; j++) {
                for (i = 0; i < N; i++) {
                    for (k = *p_sumLocal, l = *p_sumOther; k < *p_sumLocal + b; k++, l++) {
                        tarMatrix[i][k] = srcMatrix[i][l];
                    }
                }
                sumLocal += b;
                for (i = 1; i < mpiProcessNum; i++) {
                    sumOther += b;
                    rst = fMPI_Tx(&A[0][sumOther], 1, type_vec_N_b_N, i, 1, MPI_COMM_WORLD);
                    if (rst != 0)
                        return rst;
                }
                sumOther += b;
            }
            if (columnBlockNum % mpiProcessNum > 0) {
                for (i = 0; i < N; i++) {
                    for (k = *p_sumLocal, l = *p_sumOther; k < *p_sumLocal + b; k++, l++) {
                        tarMatrix[i][k] = srcMatrix[i][l];
                    }
                }
                sumLocal += b;
                for (i = 1; i < columnBlockNum % mpiProcessNum; i++) {
                    sumOther += b;
                    rst = fMPI_Tx(&A[0][sumOther], 1, type_vec_N_b_N, i, 1, MPI_COMM_WORLD);
                    if (rst != 0)
                        return rst;
                }
            }
        } else {
            sumOther = 0;
            int subBlockNum = columnBlockNum / mpiProcessNum;
            if (mpiRankID < columnBlockNum % mpiProcessNum) {
                subBlockNum++;
            }
            for (i = 0; i < subBlockNum; i++) {
                rst = fMPI_Rx(&shmA_N_Y[0][sumOther], 1, type_vec_N_b_Y, 0, 1, MPI_COMM_WORLD);
                if (rst != 0)
                    return rst;
                sumOther += b;
            }
        }
        return 0;
    }

    // Gather the final results after computation
    int GatherProcessRst() {
        return MatrixMergeOrSplit(false);
    }

public:
    // Data members used in the computation
    int N;
    int* k_prev_list;
    int* indk_next_list;

    double** A;
    double** shmA_N_Y;
    double** shmB_N_b;
    double** priC_b_b;
    double* priC_b_b_1d;
    int columnBlockNum, Y;
    int iColumnBlock, columnStart, columnEnd, rowStart, rowEnd, blockID, iProcessColumn, calBlockCnt, processColumnJ;
    MPI_Datatype type_vec_N_b_N;
    MPI_Datatype type_vec_N_b_Y;
    MPI_Datatype type_bcast_N_b_2d_calBlockCnt;
};

// Main function for MPI-based computation
int mpi_computation(double** A, int argc, char* argv[]) {
    int rst;
    CMpiCal* cMpiCal = new CMpiCal;
    rst = cMpiCal->Init(A, argc, argv);
    if (rst != 0)
        return rst;
    if (mpiRankID == 0)
        printf("Starting mpi computation...\n\n");
    struct timeval start_time, end_time;
    long seconds, microseconds;
    double elapsed;
    gettimeofday(&start_time, 0);
    rst = cMpiCal->CalRun();
    if (rst != 0)
        return rst;
    gettimeofday(&end_time, 0);
    seconds = end_time.tv_sec - start_time.tv_sec;
    microseconds = end_time.tv_usec - start_time.tv_usec;
    elapsed = seconds + 1e-6 * microseconds;
    if (mpiRankID == 0) {
		printf("**********************************MPI*****************************************\n\n");
        printf("End of MPI calculation, time consuming  calculation time: %fs\n\n", elapsed);
		printf("**********************************MPI*****************************************\n");
    }
    cMpiCal->DInit();
    delete cMpiCal;
    return 0;
}

// Main entry point of the program
int main(int argc, char* argv[]) {
    double *seqb0, *a0;
    double **seqB, **A;
    int n;
    int i, j;
    if (argc >= 2) {
        n = atoi(argv[1]);
    } else {
        n = 1;
    }
    int rst;
    size_t array2Dbytes = n * n * sizeof(double);
    printf("Start initializing the matrix...\n\n");
    seqb0 = (double*)malloc(array2Dbytes);
    a0 = (double*)malloc(array2Dbytes);
    seqB = (double**)malloc(n * sizeof(double*));
    A = (double**)malloc(n * sizeof(double*));
    for (i = 0; i < n; i++) {
        seqB[i] = seqb0 + i * n;
        A[i] = a0 + i * n;
    }
    srand(time(0));
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            A[i][j] = seqB[i][j] = (double)rand() / RAND_MAX;
        }
    }
    mpi_computation(A, argc, argv);
    if (mpiRankID == 0)
        Sequential_computation(seqB, n);
    if (mpiRankID == 0) {
        if (memcmp(seqb0, a0, array2Dbytes) == 0) {
            printf("\n\nSuccess, serial and parallel computation results are equal！\n\n\n");
        } else {
            printf("\n\nFailure, serial results do not match parallel results\n\n\n");
        }
    }
    free(seqb0);
    free(seqB);
    free(a0);
    free(A);
    return rst;
}

// Function to print the matrix
void print_matrix(double** T, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f   ", T[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");
    return;
}

// Sequential version of the Gaussian elimination algorithm
void Sequential_computation(double** a, int n) {
    int i, j, k;
    int indk;
    double c, amax;
    printf("\n\nGaussian elimination element serial computation begins ....\n\n");
    struct timeval start_time, end_time;
    long seconds, microseconds;
    double elapsed;
    gettimeofday(&start_time, 0);
    long time0;
    for (i = 0; i < n - 1; i++) {
        time0 = clock();
        amax = a[i][i];
        indk = i;
        for (k = i + 1; k < n; k++) {
            if (fabs(a[k][i]) > fabs(amax)) {
                amax = a[k][i];
                indk = k;
            }
        }
        time0 = clock();
        if (amax == 0) {
            printf("Matrix does not meet the requirements!\n");
            exit(1);
        } else if (indk != i) {
            for (j = 0; j < n; j++) {
                c = a[i][j];
                a[i][j] = a[indk][j];
                a[indk][j] = c;
            }
        }
        time0 = clock();
        for (k = i + 1; k < n; k++) {
            a[k][i] = a[k][i] / a[i][i];
        }
        time0 = clock();
        for (k = i + 1; k < n; k++) {
            c = a[k][i];
            for (j = i + 1; j < n; j++) {
                a[k][j] -= c * a[i][j];
            }
        }
    }
    gettimeofday(&end_time, 0);
    seconds = end_time.tv_sec - start_time.tv_sec;
    microseconds = end_time.tv_usec - start_time.tv_usec;
    elapsed = seconds + 1e-6 * microseconds;
	printf("**************************************************************************\n\n");
    printf("End of serial calculation, time consuming %fs\n\n", elapsed);
	printf("**************************************************************************\n");
}
