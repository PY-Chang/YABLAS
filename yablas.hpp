#pragma once
/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

// DSWAP - swap x and y
// DSCAL - x = a*x
// DCOPY - copy x into y
// DAXPY - y = a*x + y
// DDOT - dot product
// DNRM2 - Euclidean norm
// DZNRM2 - Euclidean norm
// DASUM - sum of absolute values
// IDAMAX - index of max abs value

void dswap(const int32_t N, double *DX, const int32_t incX,  double *DY, const int32_t incY);
void dscal(const int32_t N, const int32_t DA, double *DX, const int32_t incX);
void dcopy(const int32_t N, const double *DX, const int32_t incX, double *DY, const int32_t incY);
void daxpy(const int32_t N, const double DA, const double *DX, const int32_t incX, double *DY, const int32_t incY);
double ddot(const int32_t N, const double *DX, const int32_t incX, const double *DY, const int32_t incY);
double dnrm2(const int32_t N, const double *DX, const int32_t incX);
double dznrm2(const int32_t N,  std::complex<double> *DX, const int32_t incX);
double dasum(const int32_t N, const double *DX, const int32_t incX);
int32_t idamax(const int32_t N, const double *DX, const int32_t incX);


/*
 * ===========================================================================
 * Prototypes for level 2 BLAS routines
 * ===========================================================================
 */

// DGEMV - matrix vector multiply
// DSYMV - symmetric matrix vector multiply
// DTRMV - triangular matrix vector multiply
// DTRSV - solving triangular matrix problems
// DGER - performs the rank 1 operation A := alpha*x*y' + A

void dgemv( char trans, const int32_t M, const int32_t N, 
            const double alpha, const double *A, const int32_t lda,
            const double *X, const int32_t incX, const double beta,
            double *Y, const int32_t incY );
void dsymv( char Uplo,
            const int32_t N, const double alpha, const double *A,
            const int32_t lda, const double *X, const int32_t incX,
            const double beta, double *Y, const int32_t incY);
void dtrmv( char Uplo,
            char TransA, char Diag,
            const int32_t N, const double *A, const int32_t lda,
            double *X, const int32_t incX);
void dtrsv( char Uplo,
            char TransA, char Diag,
            const int32_t N, const double *A, const int32_t lda, double *X,
            const int32_t incX);
void dger( const int32_t M, const int32_t N,
                const double alpha, const double *X, const int32_t incX,
                const double *Y, const int32_t incY, double *A, const int32_t lda);


/*
 * ===========================================================================
 * Prototypes for level 3 BLAS routines
 * ===========================================================================
 */

// DGEMM - matrix matrix multiply
// DSYMM - symmetric matrix matrix multiply
// DSYRK - symmetric rank-k update to a matrix
// DTRMM - triangular matrix matrix multiply

void dgemm( char TransA,
            char TransB, const int32_t M, const int32_t N,
            const int32_t K, const double alpha, const double *A,
            const int32_t lda, const double *B, const int32_t ldb,
            const double beta, double *C, const int32_t ldc);
void dsymm( char Side,
            char Uplo, const int32_t M, const int32_t N,
            const double alpha, const double *A, const int32_t lda,
            const double *B, const int32_t ldb, const double beta,
            double *C, const int32_t ldc);

void dsyrk( char Uplo,
            char Trans, const int32_t N, const int32_t K,
            const double alpha, const double *A, const int32_t lda,
            const double beta, double *C, const int32_t ldc);

void dtrmm( char Side,
            char Uplo, char TransA,
            char Diag, const int32_t M, const int32_t N,
            const double alpha, const double *A, const int32_t lda,
            double *B, const int32_t ldb);