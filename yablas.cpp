#include <iostream>
#include <stdint.h>
#include <complex> 
#include <math.h>
#include <limits>

#include "matrix.hpp"
#include "yablas.hpp"



void dswap(const int32_t N, double *DX, const int32_t incX,  double *DY, const int32_t incY){
    if ( N <= 0 ) return;

    if ( incX ==incY ){
        for (int i = 0; i<N; i++){
            std::swap(DX[i], DY[i]);
        }
    }
    else{
        for (int x =0, y=0; x<N*incX || y<N*incY; x+=incX, y+=incY){
            std::swap(DX[x], DY[y]);
        }
    }

    for (int x =0, y=0; x<N*incX || y<N*incY; x+=incX, y+=incY){
        std::swap(DX[x], DY[y]);
    }
    // Matrix temp (0, N);
    // temp = DX;
    // DX = DY;
    // DY = temp;
    
}

void dscal(const int32_t N, const int32_t DA, double *DX, const int32_t incX){
    if ( N <= 0 ) return;
    for (int x = 0; x < N*incX; x+=incX){
        DX[x] *= DA;
    }

    // for(int i =0; i<N; i++){
    //     DX(0,i) *= DA;
    // }
}

void dcopy(const int32_t N, const double *DX, const int32_t incX, double *DY, const int32_t incY){
    if ( N <= 0 ) return;
    for (int x =0, y=0; x<N*incX || y<N*incY; x+=incX, y+=incY){
        DY[y] = DX[x];
    }
}

void daxpy(const int32_t N, const double DA, const double *DX, const int32_t incX, double *DY, const int32_t incY){
    if ( N <= 0 ) return;
    for (int x =0, y=0; x<N*incX || y<N*incY; x+=incX, y+=incY){
        DY[y] += (DA*DX[x]);
    }
    
}

double ddot(const int32_t N, const double *DX, const int32_t incX, const double *DY, const int32_t incY){
    if ( N <= 0 ) return 0;

    double result = 0;
    for (int x =0, y=0; x<N*incX || y<N*incY; x+=incX, y+=incY){
        result += DX[x]*DY[y];
    }
    return result;
}

double dnrm2(const int32_t N, const double *DX, const int32_t incX){
    if ( N <= 0 ) return 0;
    double result = 0;
    for (int x = 0; x < N*incX; x+=incX){
        result += DX[x]*DX[x];
    }
    
    return sqrt(result);
}

double dznrm2(const int32_t N,  std::complex<double> *DX, const int32_t incX){
    if ( N <= 0 ) return 0;
    double result = 0;

    for (int x = 0; x < N*incX; x+=incX){
        result += ((real(DX[x])*real(DX[x])) + (imag(DX[x])*imag(DX[x])));
    }
    return sqrt(result);
}

double dasum(const int32_t N, const double *DX, const int32_t incX){
    if ( N <= 0 ) return 0;
    double result = 0;

    for (int x = 0; x < N*incX; x+=incX){
        result += abs(DX[x]);
    }
    return result;
}

int32_t idamax(const int32_t N, const double *DX, const int32_t incX){
    if ( N <= 0 ) return -1;
    int32_t index = 0;
    double max = std::numeric_limits<double>::lowest();

    for (int x = 0; x < N*incX; x+=incX){
        if (abs(DX[x]) > max){
            max = DX[x];
            index = x;
        }
    }
    return index;
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
    double test[] = {1, 3, 3};
    std::cout<<idamax(3, test, 1);
    std::cout<<std::endl;
    for (int i = 0; i < 4 ;i++){
        
        std::cout<<b(0, i) <<std::endl;
    }

}