#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <stdint.h>
#include <math.h>


namespace py = pybind11;

#include "matrix.hpp"
// #include "yablas.hpp"

struct buffer_info {
    void *ptr;
    size_t itemsize;
    std::string format;
    int ndim;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
};


/*
 * ===========================================================================
 * Level 1 BLAS routines
 * ===========================================================================
 */


// void dswap(const int32_t N, double *DX, const int32_t incX,  double *DY, const int32_t incY){
//     if ( N <= 0 ) return;
//     if ( incX ==incY ){
//         for (int i = 0; i<N; i++){
//             std::swap(DX[i], DY[i]);
//         }
//     }
//     else{
//         for (int x =0, y=0; x<N*incX || y<N*incY; x+=incX, y+=incY){
//             std::swap(DX[x], DY[y]);
//         }
//     }
// 
//     for (int x =0, y=0; x<N*incX || y<N*incY; x+=incX, y+=incY){
//         std::swap(DX[x], DY[y]);
//     }
//     // Matrix temp (0, N);
//     // temp = DX;
//     // DX = DY;
//     // DY = temp;
// 
// }
void dswap(const int32_t N, py::array_t<double> &DX, const int32_t incX,  py::array_t<double> &DY, const int32_t incY){
    if ( N <= 0 ) return;
    
    py::buffer_info infoDX = DX.request();
    double *ptrDX = static_cast<double *>(infoDX.ptr);

    py::buffer_info infoDY = DY.request();
    double *ptrDY = static_cast<double *>(infoDY.ptr);

    // if ( incX == incY ){
    //     for (int i = 0; i<N; i++){
    //         double temp = ptrDX[i];
    //         ptrDX[i] = ptrDY[i];
    //         ptrDY[i] = temp;
    //         // std::swap(ptrDX[i], ptrDY[i]);
    //     }
    // }
    // else{
    //     for (int x =0, y=0; x<N*incX || y<N*incY; x+=incX, y+=incY){
    //         double temp = ptrDX[x];
    //         ptrDX[x] = ptrDY[y];
    //         ptrDY[x] = temp;
    //         // std::swap(DX[x], DY[y]);
    //     }
    // }

    for (int x =0, y=0; x<N*incX && y<N*incY; x+=incX, y+=incY){
        double temp = ptrDX[x];
        ptrDX[x] = ptrDY[y];
        ptrDY[y] = temp;
        // std::swap(DX[x], DY[y]);
    }
}

void dscal(const int32_t N, const int32_t DA, py::array_t<double> &DX, const int32_t incX){
    if ( N <= 0 ) return;

    py::buffer_info infoDX = DX.request();
    double *ptrDX = static_cast<double *>(infoDX.ptr);
    // std::cout<<"a at:"<ptrDX;
    for (int x = 0; x < N*incX; x+=incX){
        ptrDX[x] *= DA;
    }
}

void dcopy(const int32_t N, const py::array_t<double> &DX, const int32_t incX, py::array_t<double> &DY, const int32_t incY){
    if ( N <= 0 ) return;

    py::buffer_info infoDX = DX.request();
    double *ptrDX = static_cast<double *>(infoDX.ptr);

    py::buffer_info infoDY = DY.request();
    double *ptrDY = static_cast<double *>(infoDY.ptr);

    for (int x =0, y=0; x<N*incX || y<N*incY; x+=incX, y+=incY){
        ptrDY[y] = ptrDX[x];
    }
}

void daxpy(const int32_t N, const double DA, const py::array_t<double> &DX, const int32_t incX, py::array_t<double> &DY, const int32_t incY){
    if ( N <= 0 ) return;
    py::buffer_info infoDX = DX.request();
    double *ptrDX = static_cast<double *>(infoDX.ptr);

    py::buffer_info infoDY = DY.request();
    double *ptrDY = static_cast<double *>(infoDY.ptr);

    for (int x =0, y=0; x<N*incX || y<N*incY; x+=incX, y+=incY){
        ptrDY[y] += (DA*ptrDX[x]);
    }
    
}

double ddot(const int32_t N, const py::array_t<double> &DX, const int32_t incX, const py::array_t<double> &DY, const int32_t incY){
    if ( N <= 0 ) return 0;

    py::buffer_info infoDX = DX.request();
    double *ptrDX = static_cast<double *>(infoDX.ptr);

    py::buffer_info infoDY = DY.request();
    double *ptrDY = static_cast<double *>(infoDY.ptr);

    double result = 0;
    for (int x =0, y=0; x<N*incX || y<N*incY; x+=incX, y+=incY){
        result += ptrDX[x]*ptrDY[y];
    }
    return result;
}

double dnrm2(const int32_t N, const py::array_t<double> &DX, const int32_t incX){
    if ( N <= 0 ) return 0;

    py::buffer_info infoDX = DX.request();
    double *ptrDX = static_cast<double *>(infoDX.ptr);

    double result = 0;
    for (int x = 0; x < N*incX; x+=incX){
        result += ptrDX[x]*ptrDX[x];
    }
    
    return sqrt(result);
}

double dznrm2(const int32_t N,  py::array_t<std::complex<double>> &DX, const int32_t incX){
    if ( N <= 0 ) return 0;

    py::buffer_info infoDX = DX.request();
    std::complex<double> *ptrDX = static_cast<std::complex<double> *>(infoDX.ptr);

    double result = 0;
    for (int x = 0; x < N*incX; x+=incX){
        result += ((real(ptrDX[x])*real(ptrDX[x])) + (imag(ptrDX[x])*imag(ptrDX[x])));
    }
    return sqrt(result);
}

double dasum(const int32_t N, const py::array_t<double> &DX, const int32_t incX){
    if ( N <= 0 ) return 0;

    py::buffer_info infoDX = DX.request();
    double *ptrDX = static_cast<double *>(infoDX.ptr);

    double result = 0;
    for (int x = 0; x < N*incX; x+=incX){
        result += abs(ptrDX[x]);
    }
    return result;
}

int32_t idamax(const int32_t N, const py::array_t<double> &DX, const int32_t incX){
    if ( N <= 0 ) return -1;

    py::buffer_info infoDX = DX.request();
    double *ptrDX = static_cast<double *>(infoDX.ptr);

    int32_t index = 0;
    double max = std::numeric_limits<double>::lowest();

    for (int x = 0; x < N*incX; x+=incX){
        // std::cout<< abs(ptrDX[x]) << " ";
        if (abs(ptrDX[x]) > max){
            max = abs(ptrDX[x]);
            index = x;
        }
    }
    return index;
}


/*
 * ===========================================================================
 * Level 2 BLAS routines
 * ===========================================================================
 */

void dgemv( char trans, const int32_t M, const int32_t N, 
            const double alpha, const py::array_t<double> &A, const int32_t lda,
            const py::array_t<double> &X, const int32_t incX, const double beta,
            py::array_t<double> &Y, const int32_t incY ){

                if ((M == 0) || (N == 0) || ((alpha == 0) && (beta == 1))) return;

                py::buffer_info infoA = A.request();
                double *ptrA = static_cast<double *>(infoA.ptr);

                py::buffer_info infoX = X.request();
                double *ptrX = static_cast<double *>(infoX.ptr);

                py::buffer_info infoY = Y.request();
                double *ptrY = static_cast<double *>(infoY.ptr);
    
                if (trans == 'N' || trans == 'n'){
                    for (int i=0; i < M; i++){
                        double temp = 0;
                        for (int k=0; k<N; k++){
                            temp += ((alpha * ptrA[lda*i+k]) * ptrX[k*incX]);
                        }
                        temp += beta*ptrY[i*incY];
                        ptrY[i*incY] = temp;
                    }
                }
                else{
                    for (int i=0; i < N; i++){
                        double temp = 0;
                        for (int k=0; k<M; k++){
                            temp += ((alpha * ptrA[lda*k+i]) * ptrX[k*incX]);
                        }
                        temp += beta*ptrY[i*incY];
                        ptrY[i*incY] = temp;
                    }
                }
            }

void dsymv( char Uplo,
            const int32_t N, const double alpha, const py::array_t<double> &A,
            const int32_t lda, const py::array_t<double> &X, const int32_t incX,
            const double beta, py::array_t<double> &Y, const int32_t incY){

                if ((N == 0) || ((alpha == 0) && (beta == 1))) return;

                py::buffer_info infoA = A.request();
                double *ptrA = static_cast<double *>(infoA.ptr);

                py::buffer_info infoX = X.request();
                double *ptrX = static_cast<double *>(infoX.ptr);

                py::buffer_info infoY = Y.request();
                double *ptrY = static_cast<double *>(infoY.ptr);

                // Set up the start points in  X  and  Y.
                int kx;
                if (incX > 0) kx = 0;
                else kx = 1 - (N - 1) * incX;

                int ky;
                if (incY > 0) ky = 0;
                else ky = 1 - (N-1)*incY;

                // First form  y := beta*y.
                for (int i = 0; i < N; i++){
                    ptrY[i*incY] *= beta;
                }

                // if (alpha == 0) return;

                if (Uplo == 'U'){ // Form  y  when A is stored in upper triangle.
                    if ((incX == 1) && (incY == 1)){
                        int temp1, temp2;
                        for (int j=0; j < N; j++){
                            temp1 = alpha * ptrX[j];
                            temp2 = 0;
                            for (int i =1; i < j; i++){
                                ptrY[i] += temp1 * ptrA[i*lda+j];
                                temp2 += ptrA[i*lda+j] * ptrX[i];
                            }
                            ptrY[j] += ((temp1 * ptrA[j*lda+j]) + (alpha * temp2));
                        }
                    }
                    else{
                        int jx = kx;
                        int jy = ky;
                        int temp1, temp2;
                        int ix, iy;
                        for (int j=0; j < N; j++){
                            temp1 = alpha * ptrX[jx];
                            temp2 = 0;
                            ix = kx;
                            iy = ky;
                            for (int i=0; i < j; i++){
                                ptrY[iy] += temp1 * ptrA[i*lda+j];
                                temp2 += ptrA[i*lda+j] * ptrX[ix];
                                ix += incX;
                                iy += incY;
                            }
                            ptrY[jy] += ((temp1 * ptrA[j*lda+j]) + (alpha * temp2));
                            jx += incX;
                            jy += incY;
                        }
                    }
                }
                else{  // Form  y  when A is stored in lower triangle.
                    if ((incX == 1) && (incY == 1)){
                        int temp1, temp2;
                        for(int j = 0; j < N; j++){
                            temp1 = alpha * ptrX[j];
                            temp2 = 0;
                            ptrY[j] += temp1 * ptrA[j*lda+j];
                            for(int i = j+1; i < N; i++){
                                ptrY[i] += temp1 * ptrA[i*lda+j];
                                temp2 += ptrA[i*lda+j] * ptrX[i];
                            }
                            ptrY[j] += alpha * temp2;
                        }
                    }
                    else{
                        int jx = kx;
                        int jy = ky;
                        int temp1, temp2;
                        int ix, iy;
                        for(int j = 0; j < N; j++){
                            temp1 = alpha * ptrX[jx];
                            temp2 = 0;
                            ptrY[jy] += temp1 * ptrA[j*lda+j];
                            ix = jx;
                            iy = jy;
                            for (int i = j+1; i < N; i++){
                                ix += incX;
                                iy += incY;
                                ptrY[iy] += temp1 * ptrA[i*lda+j];
                                temp2 += ptrA[i*lda+j] * ptrX[ix];
                            }
                            ptrY[jy] += alpha * temp2;
                            jx += incX;
                            jy += incY;
                        }
                    }
                }
            }

void dtrmv( char Uplo,
            char TransA, char Diag,
            const int32_t N, const py::array_t<double> &A, const int32_t lda,
            py::array_t<double> &X, const int32_t incX){
            if (N == 0) return;

            py::buffer_info infoA = A.request();
            double *ptrA = static_cast<double *>(infoA.ptr);

            py::buffer_info infoX = X.request();
            double *ptrX = static_cast<double *>(infoX.ptr);

            bool NOUNIT = (Diag == 'U');
            int KX;
            if (incX <= 0){
                KX = 1 - (N - 1) * incX;
            }
            else if (incX != 1){
                KX = 0;
            }

            if (TransA == 'N'){
                if (Uplo == 'U'){
                    if (incX == 1){
                        int TEMP = 0;
                        for(int J = 0 ;J < N; J++){
                            if (ptrX[J] != 0){
                                TEMP = ptrX[J];
                                ptrX[J] = 0;
                                for (int I = 0; I <= J; I++){
                                    ptrX[I] += TEMP * ptrA[I*lda+J];
                                }   
                                if (NOUNIT){
                                    ptrX[J] = ptrX[J] * ptrA[J*lda+J];
                                }
                            }
                        }     
                    }
                    else{
                        int JX = KX;
                        int TEMP, IX;
                        for(int J = 0 ;J < N; J++){
                            if (ptrX[JX] != 0){
                                TEMP = ptrX[JX];
                                ptrX[JX] = 0;
                                IX = KX;
                                for (int I = 0; I <= J ; I++){
                                    ptrX[IX] = ptrX[IX] + TEMP * ptrA[I*lda+J];
                                    IX += incX;
                                }
                                if (NOUNIT)
                                    ptrX[JX] = ptrX[JX] * ptrA[J*lda+J];
                            }
                            JX += incX;
                        }
                    }
                }
                else{
                    if (incX == 1){
                        int TEMP;
                        for (int J = N-1; J > -1; J--){
                            if (ptrX[J] != 0){
                                TEMP = ptrX[J];
                                ptrX[J] = 0;
                                for (int I = N-1; I > J-1; I--){
                                    ptrX[I] += TEMP * ptrA[I*lda+J];
                                }  
                                if (NOUNIT)
                                    ptrX[J] = ptrX[J] * ptrA[J*lda+J];
                            }
                        }      
                    }
                    else{
                        KX += (N - 1) * incX;
                        int JX = KX;
                        int TEMP, IX;
                        for (int J = N-1; J > -1; J--){
                            if (ptrX[JX] != 0){
                                TEMP = ptrX[JX];
                                ptrX[JX] = 0;
                                IX = KX;
                                for (int I = N-1; I > J-1; I--){
                                    ptrX[IX] = ptrX[IX] + TEMP * ptrA[I*lda+J];
                                    IX -= incX;
                                }
                                if (NOUNIT)
                                    ptrX[JX] = ptrX[JX] * ptrA[J*lda+J];
                            }
                            JX -= incX;
                        }
                    }
                }
            }
            else{
                //  Form  x := A**T*x.
                if (Uplo == 'U'){
                    if (incX == 1){
                        int TEMP;
                        for (int J = N-1; J > -1; J--){
                            TEMP = 0;
                            if (NOUNIT)
                                TEMP *= ptrA[J*lda+J];
                            for(int I = J; I > -1; I--){
                                TEMP += ptrA[I*lda+J] * ptrX[I];
                            }
                            ptrX[J] = TEMP;
                        }
                    }
                    else{
                        int JX = KX + (N - 1) * incX;
                        int TEMP, IX;
                        for (int J = N-1; J > -1; J--){
                            TEMP = 0;
                            IX = JX;
                            if (NOUNIT)
                                TEMP *= ptrA[J*lda+J];
                            for(int I = J; I > -1; I--){
                                TEMP += ptrA[I*lda+J] * ptrX[IX];
                                IX -= incX;
                            }
                            ptrX[JX] = TEMP;
                            JX -= incX;
                        }
                    }
                }
                else{
                    if (incX == 1){
                        int TEMP;
                        for (int J = 0; J < N; J++){
                            TEMP = 0;
                            if (NOUNIT)
                                TEMP *= ptrA[J*lda+J];
                            for (int I = J; I < N; I++){
                                TEMP += ptrA[I*lda+J] * ptrX[I];
                            }
                            ptrX[J] = TEMP;
                        }
                    }
                    else{
                        int JX = KX;
                        int TEMP, IX;
                        for (int J = 0; J < N; J++){
                            TEMP = 0;
                            IX = JX;
                            if (NOUNIT)
                                TEMP *= ptrA[J*lda+J];
                            for (int I = J; I < N; I++){
                                TEMP += ptrA[I*lda+J] * ptrX[IX];
                                IX += incX;
                            }
                            ptrX[JX] = TEMP;
                            JX += incX;
                        }
                    }
                }
            }
        }

void dtrsv( char Uplo,
            char TransA, char Diag,
            const int32_t N, const py::array_t<double> &A, const int32_t lda, py::array_t<double> &X,
            const int32_t incX){

            if (N == 0) return;

            py::buffer_info infoA = A.request();
            double *ptrA = static_cast<double *>(infoA.ptr);

            py::buffer_info infoX = X.request();
            double *ptrX = static_cast<double *>(infoX.ptr);

            bool NOUNIT = (Diag == 'N');
            int KX;
            if (incX <= 0){
                KX = 1 - (N - 1) * incX;
            }
            else if (incX != 1){
                KX = 0;
            }

            if (TransA == 'N'){
            // #
            // #        Form  x := inv( A )*x.
            // #
                if (Uplo == 'U'){
                    if (incX == 1){
                        int TEMP;
                        for (int J = N-1; J > -1; J--){
                            if (ptrX[J] != 0){
                                if (NOUNIT)
                                    ptrX[J] = ptrX[J] / ptrA[J*lda+J];
                                TEMP = ptrX[J];
                                for (int I = J-1; I > -1; I--){
                                    ptrX[I] -= TEMP * ptrA[I*lda+J];
                                }
                            }
                        }
                    }
                    else{
                        int JX, TEMP, IX;
                        JX = KX + (N - 1) * incX;
                        for (int J = N-1; J > -1; J--){
                            if (ptrX[JX] != 0){
                                if (NOUNIT)
                                    ptrX[JX] = ptrX[JX] / ptrA[J*lda+J];
                                TEMP = ptrX[JX];
                                IX = JX;
                                for (int I = J-1; I > -1; I--){
                                    IX -= incX;
                                    ptrX[IX] -= TEMP * ptrA[I*lda+J];
                                }
                            }
                            JX -= incX;
                        }
                    }
                }
                else{
                    if (incX == 1){
                        int TEMP;
                        for (int J = 0; J < N; J++){
                            if (ptrX[J] != 0){
                                if (NOUNIT)
                                    ptrX[J] = ptrX[J] / ptrA[J*lda+J];
                                TEMP = ptrX[J];
                                for (int I = J+1; I < N; I++)
                                    ptrX[I] -= TEMP * ptrA[I*lda+J];
                            }
                        }
                    }
                    else{
                        int JX = KX;
                        int TEMP, IX;
                        for (int J = 0; J < N; J++){
                            if (ptrX[JX] != 0){
                                if (NOUNIT)
                                    ptrX[JX] = ptrX[JX] / ptrA[J*lda+J];
                                TEMP = ptrX[JX];
                                IX = JX;
                                for (int I = J+1; I < N; I++){
                                    IX += incX;
                                    ptrX[IX] -= TEMP * ptrA[I*lda+J];
                                }
                            }
                            JX += incX;
                        }
                    }
                }
            }
            else{
                // # Form  x := inv( A**T )*x.
                if (Uplo == 'U'){
                    if (incX == 1){
                        int TEMP;
                        for (int J = 0; J < N; J++){
                            TEMP = ptrX[J];
                            for (int I = 0; I < J; I++)
                                TEMP -= ptrA[I*lda+J] * ptrX[I];
                            if (NOUNIT)
                                TEMP = TEMP / ptrA[J*lda+J];
                            ptrX[J] = TEMP;
                        }
                    }
                    else{
                        int JX = KX;
                        int TEMP, IX;
                        for (int J = 0; J < N; J++){
                            TEMP = ptrX[JX];
                            IX = KX;
                            for (int I = 0; I < J; I++){
                                TEMP -= ptrA[I*lda+J] * ptrX[IX];
                                IX += incX;
                            }
                            if (NOUNIT)
                                TEMP = TEMP / ptrA[J*lda+J];
                            ptrX[JX] = TEMP;
                            JX += incX;
                        }
                    }
                }
                else{
                    if (incX == 1){
                        int TEMP;
                        for (int J = N -1; J > -1; J--){
                            TEMP = ptrX[J];
                            for (int I = N-1; I > J; I--)
                                TEMP -= ptrA[I*lda+J] * ptrX[I];
                            if (NOUNIT)
                                TEMP = TEMP / ptrA[J*lda+J];
                            ptrX[J] = TEMP;
                        }
                    }
                    else{
                        KX += (N - 1) * incX;
                        int JX = KX;
                        int TEMP, IX;
                        for (int J = N -1; J > -1; J--){
                            TEMP = ptrX[JX];
                            IX = KX;
                            for (int I = N-1; I > J; I--){
                                TEMP -= ptrA[I*lda+J] * ptrX[IX];
                                IX -= incX;
                            }
                            if (NOUNIT)
                                TEMP = TEMP / ptrA[J*lda+J];
                            ptrX[JX] = TEMP;
                            JX -= incX;
                        }
                    }
                }
            }
        }

void dger(  const int32_t M, const int32_t N,
            const double alpha, const py::array_t<double> &X, const int32_t incX,
            const py::array_t<double> &Y, const int32_t incY, py::array_t<double> &A, const int32_t lda){

                if ((M == 0) || (N == 0) || (alpha == 0)) return;

                py::buffer_info infoX = X.request();
                double *ptrX = static_cast<double *>(infoX.ptr);

                py::buffer_info infoY = Y.request();
                double *ptrY = static_cast<double *>(infoY.ptr);

                py::buffer_info infoA = A.request();
                double *ptrA = static_cast<double *>(infoA.ptr);

                int jy;
                if (incY > 0){
                    jy =0;
                }
                else{
                    jy = 0 - (N-1)*incY;
                }

                if (incX == 1){
                    int temp;
                    for (int j = 0; j < N; j++){
                        if (ptrY[jy] != 0){
                            temp = alpha*ptrY[jy];
                            for (int i = 0; i < M; i++){
                                ptrA[i*lda+j] += ptrX[i]*temp;
                            }
                        }
                        jy += incY;
                    }
                }
                else{
                    int kx;
                    if (incX > 0) kx = 0;
                    else kx = 0 - (M-1)*incX;

                    int temp, ix;
                    for (int j = 0; j < N; j++){
                        if (ptrY[jy] != 0){
                            temp = alpha*ptrY[jy];
                            ix = kx;
                            for (int i = 0; i < M; i++){
                                ptrA[i*lda+j] += ptrX[ix]*temp;
                                ix += incX;
                            }
                        }
                        jy += incY;
                    }
                }
                // // std::cout<<ptrA<<std::endl;
                // for(int i =0; i < lda; i++){
                //     for (int j =0; j<lda; j++){
                //         std::cout<<ptrA[i*lda+j]<<" ";
                //     }
                //     std::cout<<std::endl;
                // }
            }


/*
 * ===========================================================================
 * Level 3 BLAS routines
 * ===========================================================================
 */

void dgemm( char TransA,
            char TransB, const int32_t M, const int32_t N,
            const int32_t K, const double alpha, const py::array_t<double> &A,
            const int32_t lda, const py::array_t<double> &B, const int32_t ldb,
            const double beta, py::array_t<double> &C, const int32_t ldc){

                if ((M == 0) || (N == 0) || (((alpha == 0) || (K == 0)) && (beta == 1))) return;

                // And if  alpha.eq.zero.
                if (alpha == 0){
                    if (beta == 0){
                        // for (int i = 0; i < )
                    }
                }

                py::buffer_info infoA = A.request();
                double *ptrA = static_cast<double *>(infoA.ptr);

                py::buffer_info infoB = B.request();
                double *ptrB = static_cast<double *>(infoB.ptr);

                py::buffer_info infoC = C.request();
                double *ptrC = static_cast<double *>(infoC.ptr);

                for (int i = 0; i < M; i++){
                    for (int j = 0; j < N; j++){
                        for (int k = 0; k < N; k++){
                            ptrC[i*ldc+j] *= beta;
                        }
                    }
                }

                for (int i = 0; i < M; i++){
                    for (int j = 0; j < N; j++){
                        for (int k = 0; k < N; k++){
                            ptrC[i*ldc+j] += ptrA[i*lda+k] * ptrB[j*ldb+k];
                        }
                    }
                }

                for (int i = 0; i < M; i++){
                    for (int j = 0; j < N; j++){
                        for (int k = 0; k < N; k++){
                            ptrC[i*ldc+j] *= alpha;
                        }
                    }
                }




            }


void dtrsm( char Side,
            char Uplo, char TransA,
            char Diag, const int32_t M, const int32_t N,
            const double alpha, const py::array_t<double> &A, const int32_t lda,
            py::array_t<double> &B, const int32_t ldb){

                // Quick return if possible.
                if ((M == 0) || (N == 0)) return;

                py::buffer_info infoA = A.request();
                double *ptrA = static_cast<double *>(infoA.ptr);

                py::buffer_info infoB = B.request();
                double *ptrB = static_cast<double *>(infoB.ptr);

                bool lside = (Side == 'L');
                int nrowa = 0;
                if (lside) nrowa = M;
                else nrowa = N;

                bool nounit = (Diag == 'N');
                bool upper = (Uplo == 'U');

                if (alpha == 0){
                    for (int j = 0; j < N; j++){
                        for (int i = 0; i < M; i++){
                            ptrB[i*ldb+j] = 0;
                        }
                    }
                }

                // Start the operations.
                if (lside){
                    if (TransA == 'N'){
                        // Form  B := alpha*inv( A )*B.
                        if (upper){
                            for (int j = 0; j < N; j++){
                                if (alpha != 1){
                                    for (int i=0; i < M; i++){
                                        ptrB[i*ldb+j] *= alpha;
                                    }
                                }
                                for (int k = M-1; k > -1; k--){
                                    if (ptrB[k*ldb+j] != 0){
                                        if (nounit) ptrB[k*ldb+j] /= ptrA[k*lda+k];
                                        for (int i = 0; i < k; i++){
                                            ptrB[i*ldb+j] -= (ptrB[k*ldb+j]*ptrA[i*lda+k]);
                                        }
                                    }
                                }
                            }
                        }
                        else{
                            for (int j = 0; j < N; j++){
                                if (alpha != 1){
                                    for (int i= 0; i < M; i++){
                                        ptrB[i*ldb+j] *= alpha;
                                    }
                                }
                                for (int k = 0; k < M; k++){
                                    if (ptrB[k*ldb+j] != 0){
                                        if (nounit) ptrB[k*ldb+j] /= ptrA[k*lda+k];
                                        for (int i = k+1; i < M; i++){
                                            ptrB[i*ldb+j] -= (ptrB[k*ldb+j]*ptrA[i*lda+k]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else{ //Form  B := alpha*inv( A**T )*B.
                        if (upper){
                            double temp;
                            for(int j = 0; j < N; j++){
                                for (int i = 0; i < M; i++){
                                    temp = alpha * ptrB[i*ldb+j];
                                    for (int k = 0; k < i; k++){
                                        temp -= ptrA[k*lda+i] * ptrB[k*ldb+j];
                                    }
                                    if (nounit) temp /= ptrA[i*lda+i];
                                    ptrB[i*lda+j] = temp;
                                }
                            }
                        }
                        else{
                            double temp;
                            for (int j = 0; j < N; j++){
                                for (int i = M-1; i > -1; i--){
                                    temp =  alpha * ptrB[i*ldb+j];
                                    for (int k = i+1; k < M; k++){
                                        temp -= ptrA[k*lda+i] * ptrB[k*ldb+j];
                                    }
                                    if (nounit) temp /= ptrA[i*lda+i];
                                    ptrB[i*lda+j] = temp;
                                }
                            }
                        }
                    }
                }
                else{
                    if (TransA == 'N'){
                        if (upper){
                            for (int j = 0; j < N; j++){
                                if (alpha != 1){
                                    for (int i = 0; i < M; i++){
                                        ptrB[i*ldb+j] *= alpha;
                                    }
                                }
                                for (int k = 0; k < j; k++){
                                    if (ptrA[k*lda+j] != 0){
                                        for (int i = 0; i < M; i++){
                                            ptrB[i*ldb+j] -= ptrA[k*lda+j]*ptrB[i*ldb+k];
                                        }
                                    }
                                }
                                if (nounit){
                                    double temp = 1/ptrA[j*lda+j];
                                    for (int i = 0; i < M; i++){
                                        ptrB[i*ldb+j] *= temp;
                                    }
                                }
                            }
                        }
                        else{
                            for (int j = N-1; j > -1; j--){
                                if (alpha != 1){
                                    for (int i = 0; i < M; i++){
                                        ptrB[i*ldb+j] *= alpha;
                                    }
                                }
                                for(int k = j+1; k < N; k++){
                                    if (ptrA[k*lda+j] != 0){
                                        for (int i = 0; i < M; i++){
                                            ptrB[i*ldb+j]-= ptrA[k*lda+j]*ptrB[i*ldb+k];
                                        }
                                    }
                                }
                                if (nounit){
                                    double temp = 1/ptrA[j*lda+j];
                                    for (int i = 0; i < M; i++){
                                        ptrB[i*ldb+j] *= temp;
                                    }
                                }
                            }
                        }
                    }
                    else{
                        // Form  B := alpha*B*inv( A**T ).
                        if (upper){
                            for (int k = N -1; k > -1; k--){
                                if (nounit){
                                    double temp = 1/ptrA[k*lda+k];
                                    for (int i = 0; i < M; i++){
                                        ptrB[i*ldb+k] *= temp;
                                    }
                                }
                                for (int j = 0; j < k; j++){
                                    if (ptrA[j*lda+k] != 0){
                                        double temp = ptrA[j*lda+k];
                                        for (int i = 0; i < M; i++){
                                            ptrB[i*ldb+j] -= temp*ptrB[i*ldb+k];
                                        }
                                    }
                                }
                                if (alpha != 1){
                                    for (int i = 0; i< M; i++){
                                        ptrB[i*ldb+k] *= alpha;
                                    }
                                }
                            }
                        }
                        else{
                            for (int k = 0; k < N; k++){
                                if (nounit){
                                    double temp = 1/ptrA[k*lda+k];
                                    for (int i = 0; i < M; i++){
                                        ptrB[i*ldb+k] *= temp;
                                    }
                                }
                                for (int j = k+1; j < N; j++){
                                    if (ptrA[j*lda+k] != 0){
                                        double temp = ptrA[j*lda+k];
                                        for (int i = 0; i < M; i++){
                                            ptrB[i*ldb+j] -= temp*ptrB[i*ldb+k];
                                        }

                                    }
                                }
                                if (alpha != 1){
                                    for (int i = 0; i < M; i++){
                                        ptrB[i*ldb+k] *= alpha;
                                    }
                                }
                            }
                        }
                    }
                }
            }


int main (int argc, char *argv[]){
    Matrix a(1, 4);
    Matrix b(1, 4);
    int count = 1;
    for (int i = 0; i < 4 ;i++){
        
            a(0, i) = ++count;
    }

    for (int i = 0; i < 4 ;i+=1){
        b(0,i) =-1;
    }
    // dswap(4, a, 1, b, 1);
    // dscal(4, 5, b.buffer(), 1);
    // dcopy(4, a.buffer(), 1, b.buffer(), 1);
    // daxpy(4, 9, a.buffer(), 1, b.buffer(), 1);
    // std::cout<<ddot(4, a.buffer(), 1, b.buffer(), 1);
    // std::cout<<dnrm2(4, b.buffer(), 1);
    // std::complex<double> c[3] = {{1, 2}, {2, 3}, {3, 4}};
    // std::cout<<dznrm2(3, c, 1);

    double A[] = {1, 2, 3, 0, 1, 4, 0, 0, 1}; // Up
    // double A[] = {1, 0, 0, 2, 1, 0, 3, 4, 1}; // low
    // double A[] = {1, 2, 4, 0 ,6, 7, 0, 0, 9}; // non unit up
    // double A[] = {1, 0, 0, 2, 3, 0, 4, 5, 8}; // non unit low
    double X[] = {14, 14, 3}; // up
    // double X[] = {14, 0, 14, 0, 3};
    // double X[] = {1 ,4 ,14}; // low
    // double X[] = {1, 0, 4, 0, 14};
    // double X[] = {17, 21 ,24};
    double Y[] = {2, 0, 2, 0, 2};
    // dsymv('L', 3, 1, A, 3, X, 2, 3, Y, 2);
    // dtrsv('U', 'N', 'U', 3, A, 3, X, 1);

    // dger(3, 3, 2, Y, 2, Y, 2, A, 3);
    // std::cout<<idamax(3, test, 1);
    // std::cout<<std::endl;
    // for (int i = 0; i < 3 ;i++){
    //     for (int j = 0; j < 3; j++)
    //         std::cout<<A[i*3+j]<<" ";
    //     std::cout<<std::endl;
    // }

    for (int i = 0; i < 3 ;i++){
        std::cout<<X[i]<<" ";
    }

}

PYBIND11_MODULE(yablas, m){
    // Level 1
    m.def("dswap", &dswap);
    m.def("dscal", &dscal);
    m.def("dcopy", &dcopy);
    m.def("daxpy", &daxpy);
    m.def("ddot", &ddot);
    m.def("dnrm2", &dnrm2);
    m.def("dznrm2", &dznrm2);
    m.def("dasum", &dasum);
    m.def("idamax", &idamax);

    // Level 2
    m.def("dgemv", &dgemv);
    m.def("dsymv", &dsymv);
    m.def("dtrmv", &dtrmv);
    m.def("dtrsv", &dtrsv, py::return_value_policy:: reference);
    m.def("dger", &dger, py::return_value_policy:: reference);

    // Level 3
    m.def("dgemm", &dgemm);
    m.def("dtrsm", &dtrsm);
}