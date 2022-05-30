#include <iostream>
#include <thread>
#include <math.h>
#include <time.h>
#include "matrix.hpp"

double *naive(double *tempp, const double *o, const double *p, const int m){

    double sum=0;
    for(int i=0; i<m; i++) {
        for(int j=0; j<m; j++) {
            tempp[i*m+j] = 0;
        }
    }

    for(int k=0; k<m; ++k){
        for(int i=0; i<m; ++i){
            // sum = o[i*m+k];
            for(int j=0; j<m; ++j)
                tempp[i*m+j] += o[i*m+k] * p[k*m+j];
        }
    }

    // int tileSize = 8;
    // for (int i = 0; i < m; i += tileSize){
    //     for (int j = 0; j < m; j += tileSize){
    //         for (int k = 0; k < m; k += tileSize ){
    //             //int xRange = std::min(i+tileSize, mrow);
    //             int xRange = ((i+tileSize) < m) ? (i+tileSize) : m;
    //             for (int x = i; x < xRange; x++){
    //                 //int yRange = std::min(j+tileSize, mcol);
    //                 int yRange = ((j+tileSize) < m) ? (j+tileSize) : m;
    //                 for (int y = j; y < yRange; y++){
    //                     //int zRange = std::min(k+tileSize, mrow);
    //                     int zRange = ((k+tileSize) < m) ? (k+tileSize) : m;
    //                     for (int z = k; z < zRange; z++){
    //                         //result(x, y) += mat1(x, z) * mat2(z, y);
    //                         tempp[x*m + y] += o[x*m + z] * p[z*m + y];
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    return tempp;
}

double *add_matrix(double *add, const double *m, const double *n, const int x, const int y){

    for(int i=0; i<x; i++) {
        for(int j=0; j<y; j++) {
            add[i*y+j] = m[i*y+j] + n[i*y+j];
        }
    }
    
    return add;
}

double *sub_matrix(double *sub, const double *m, const double *n, const int x, const int y){

    for(int i=0; i<x; i++) {
        for(int j=0; j<y; j++) {
            sub[i*y+j] = m[i*y+j] - n[i*y+j];
        }
    }
    
    return sub;
}

double *strassen(double *temp, const double *a, const double *b, const int n){

    if(n <= 128)                    // 設定閥值，決定切到多小時換回最原始的演算法
        return naive(temp, a, b, n);
    // if (n <= 128){
    //     for(int k=0; k<n; ++k){
    //         for(int i=0; i<n; ++i){
    //             // sum = o[i*m+k];
    //             for(int j=0; j<n; ++j)
    //                 temp[i*n+j] += a[i*n+k] * b[k*n+j];
    //         }
    //     }
    //     return temp;
    // }
    
    int n2 = n/2;
    long long  reserved_size = n2*n2;
    double *p1 = (double*)malloc(sizeof(double)*reserved_size);
    double *p2 = (double*)malloc(sizeof(double)*reserved_size);
    double *p3 = (double*)malloc(sizeof(double)*reserved_size);    
    double *p4 = (double*)malloc(sizeof(double)*reserved_size);
    double *p5 = (double*)malloc(sizeof(double)*reserved_size);
    double *p6 = (double*)malloc(sizeof(double)*reserved_size);
    double *p7 = (double*)malloc(sizeof(double)*reserved_size);
    // double p1[reserved_size], p2[reserved_size], p3[reserved_size], p4[reserved_size];
    // double p5[reserved_size], p6[reserved_size], p7[reserved_size];

    double *a11 = (double*)malloc(sizeof(double)*reserved_size);
    double *a12 = (double*)malloc(sizeof(double)*reserved_size);
    double *a21 = (double*)malloc(sizeof(double)*reserved_size);
    double *a22 = (double*)malloc(sizeof(double)*reserved_size);
    // double a11[reserved_size], a12[reserved_size];
    // double a21[reserved_size], a22[reserved_size];

    double *b11 = (double*)malloc(sizeof(double)*reserved_size);
    double *b12 = (double*)malloc(sizeof(double)*reserved_size);
    double *b21 = (double*)malloc(sizeof(double)*reserved_size);
    double *b22 = (double*)malloc(sizeof(double)*reserved_size);    
    // double b11[reserved_size], b12[reserved_size];
    // double b21[reserved_size], b22[reserved_size];

    double *m11 = (double*)malloc(sizeof(double)*reserved_size);
    double *m12 = (double*)malloc(sizeof(double)*reserved_size);
    double *m21 = (double*)malloc(sizeof(double)*reserved_size);
    double *m22 = (double*)malloc(sizeof(double)*reserved_size);  
    // double m11[reserved_size], m12[reserved_size];
    // double m21[reserved_size], m22[reserved_size];

    double *a_temp = (double*)malloc(sizeof(double)*reserved_size);
    double *b_temp = (double*)malloc(sizeof(double)*reserved_size);
    // double a_temp[reserved_size], b_temp[reserved_size];

    for (int i = 0; i < n2; ++i){        // 分割矩陣，將 A、B 均分為四塊(即 11,12,21,22)
        for (int j = 0; j < n2; ++j){
            a11[i*n2+j] = a[i*n+j];
            a12[i*n2+j] = a[i*n+n2+j];
            a21[i*n2+j] = a[(i+n2)*n+j];
            a22[i*n2+j] = a[(i+n2)*n+j+n2];
            b11[i*n2+j] = b[i*n+j];
            b12[i*n2+j] = b[i*n+n2+j];
            b21[i*n2+j] = b[(i+n2)*n+j];
            b22[i*n2+j] = b[(i+n2)*n+j+n2];
        }
    }

    //p1 = (b21+b22)＊(a12-a22)  >>繼續 call strassen 分割直到其 n 低於閥值就可以去矩陣相乘，即可算出 p1(用 original_mul)
    std::thread t1(strassen,p1, add_matrix(b_temp, b21, b22,n2,n2), sub_matrix(a_temp, a12, a22,n2,n2), n2);

    //p2 = (a11+a22)*(b11+b22)
    std::thread t2(strassen, p2, add_matrix(a_temp, a11, a22,n2,n2), add_matrix(b_temp, b11, b22,n2,n2), n2);

    //p3 = (b11+b12)*(a11-a21)
    std::thread t3(strassen, p3, add_matrix(b_temp, b11, b12,n2,n2), sub_matrix(a_temp, a11, a21,n2,n2), n2);
    
    //p4 = (a11+a12)*b22
    std::thread t4(strassen, p4, add_matrix(a_temp, a11, a12,n2,n2), b22, n2);
    
    //p5 = a11*(b12-b22)
    std::thread t5(strassen, p5, a11, sub_matrix(b_temp, b12, b22,n2,n2), n2);
    
    //p6 = a22*(b21-b11)
    std::thread t6(strassen, p6, a22, sub_matrix(b_temp, b21,b11,n2,n2), n2);
    
    //p7 = (a21+a22)*b11
    std::thread t7(strassen, p7, add_matrix(a_temp,a21,a22,n2,n2), b11, n2);

    t1.detach();
    t2.detach(); 
    t3.detach();  
    t4.detach(); 
    t5.detach(); 
    t6.detach(); 
    t7.detach(); 

    // t1.join();
    // t2.join(); 
    // t3.join();  
    // t4.join(); 
    // t5.join(); 
    // t6.join(); 
    // t7.join(); 

    //m11 = (p1+p2)-(p4+p6) >> 用上面做出來的結果去算各個 m
    std::thread tt1(sub_matrix, m11, add_matrix(a_temp, p1, p2, n2,n2), sub_matrix(b_temp, p4, p6, n2,n2), n2, n2);
    //m12 =  p4+p5
    std::thread tt2(add_matrix, m12, p4, p5, n2, n2);
    //m21 = p6+p7
    std::thread tt3(add_matrix, m21, p6, p7, n2, n2);
    //m22 = (p2+p5)-(p3+p6)
    std::thread tt4(sub_matrix, m22, add_matrix(a_temp, p2, p5, n2, n2), add_matrix(b_temp, p3, p6, n2, n2), n2, n2);

    tt1.detach();
    tt2.detach(); 
    tt3.detach();  
    tt4.detach(); 

    for(int i=0; i<n2; ++i){            //紀錄所算出來的 m
        for(int j=0; j<n2; ++j){
            temp[i*n+j] = m11[i*n2+j];
            temp[i*n+j+n2] = m12[i*n2+j];
            temp[(i+n2)*n+j] = m21[i*n2+j];
            temp[(i+n2)*n+j+n2] = m22[i*n2+j];
        }
    }

    return temp;
}


int main(int argc, char *argv[]) {
    int N=0, original_N=0;

    std::cout<<"Enter a number N.\n";
    std::cin>>original_N;
    N = original_N;

    bool illegal_N = false;
    if (original_N & (original_N-1))  // check whether original_N is 2^n
    {
        int temp = log2(original_N);
        N = pow(2,(temp+1));      // 把原本的 N 擴大成離他最近的 2^n
        illegal_N = true;
    }

    int size = N;
    Matrix A(N, N);
    int count = 1;
    for (int i = 0; i < N ;i++){
        for (int j = 0; j< N; j++){
            A(i, j) = count++;
        }
    }
    Matrix B = A;


    double *a = A.buffer();
    double *b = B.buffer();

    // double mid[N*N];
    double *mid = (double*)malloc(sizeof(double)*(N*N));

    // clock_t start, finish;
    // start = clock();
    // double *stra = strassen(mid, a, b, N);
    // finish = clock();

    // std::cout << (double)(finish - start) / CLOCKS_PER_SEC;
    // std::cout<<"\n";
    
    // std::cout.setf(std::ios::fixed);

    struct timespec start, finish;
    double elapsed;

    clock_gettime(CLOCK_MONOTONIC, &start);

    double *stra = strassen(mid, a, b, N);

    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    std::cout<<elapsed<<std::endl;
    // double *naiv = naive(mid, a, b, N);

    // for(int i=0; i<N; i++) {
    //     for(int j=0; j<N; j++) {
    //         std::cout<<stra[i*N+j]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
    // std::cout<<std::endl;
    // for(int i=0; i<N; i++) {
    //     for(int j=0; j<N; j++) {
    //         std::cout<<naiv[i*N+j]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
    free(mid);

    return 0;
}