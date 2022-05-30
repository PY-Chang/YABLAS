# YABLAS
Yet Another blas library based on C++ and Python hybrid system. 

## Usage
* `git clone https://github.com/PY-Chang/YABLAS.git`
* `make`
* (in python) `import yablas`

## Supported Subroutines

### Level 1
* DSWAP - swap x and y
* DSCAL - x = a*x
* DCOPY - copy x into y
* DAXPY - y = a*x + y
* DDOT - dot product
* DNRM2 - Euclidean norm
* DZNRM2 - Euclidean norm
* DASUM - sum of absolute values
* IDAMAX - index of max abs value


### Level 2
* DGEMV - matrix vector multiply
* DSYMV - symmetric matrix vector multiply
* DTRMV - triangular matrix vector multiply
* DTRSV - solving triangular matrix problems
* DGER - performs the rank 1 operation A := alpha*x*y' + A

### Level 3
* DGEMM - matrix matrix multiply
* DSYMM - symmetric matrix matrix multiply
* DSYRK - symmetric rank-k update to a matrix
* DTRMM - triangular matrix matrix multiply
