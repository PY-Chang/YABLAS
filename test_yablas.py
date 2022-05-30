from decimal import InvalidContext
from distutils.command.upload import upload
from locale import DAY_1
import sys

from pyparsing import alphas
# sys.path.append(sys.path.__dir__)
import yablas
import time
import numpy as np

import pytest


# ===========================================================================
# test for level 1 routines
# ===========================================================================

class TestLevel1:
    def test_dswap(self):
        N = 3
        DX = np.ones(shape=(3,1))
        incX = 1
        DY = np.zeros(shape=(3, 1))
        incY = 1
        yablas.dswap(N, DX, incX, DY, incY)

        assert np.array_equal(DX, np.array([[0], [0], [0]], dtype = np.float64), equal_nan=False)
        assert np.array_equal(DY, np.array([[1], [1], [1]], dtype = np.float64), equal_nan=False)

    def test_dscal(self):
        N = 3
        DA = 5
        DX = np.ones(shape=(3, 1), dtype = np.float64)
        incX = 1
        yablas.dscal(N, DA, DX, incX)

        # assert DX.all() == np.array([5, 5, 5]).all()
        # print(DX)
        assert np.array_equal(DX, np.array([[5], [5], [5]], dtype = np.float64), equal_nan=False)

    def test_dcopy(self):
        N = 5
        DX = np.ones(shape=(1,5)) + 3
        incX = 1
        DY = np.zeros(shape=(1, 5))
        incY = 1
        yablas.dcopy(N, DX, incX, DY, incY)

        # assert DX.all() == DY.all()
        assert np.array_equal(DX, DY, equal_nan=False)

    def test_daxpy(self):
        N = 5
        DA = 2
        DX = np.ones(shape=(1,5))
        incX = 1
        DY = np.ones(shape=(1,5)) + 10.0
        incY = 1
        yablas.daxpy(N, DA, DX, incX, DY, incY)

        # assert DY.all() == np.array([13, 13, 13, 13, 13).all()
        assert np.array_equal(DY, np.array([[13, 13, 13, 13, 13]], dtype = np.float64), equal_nan=False)

    def test_ddot(self):
        N = 10
        DX = np.ones(shape=(1,10))
        incX = 1
        DY = np.ones(shape=(1,10)) + 5
        incY = 1
        dot = yablas.ddot(N, DX, incX, DY, incY)

        assert dot == 60

    def test_ddot_quick_return(self):
        N = 0
        DX = np.ones(shape=(1,10))
        incX = 1
        DY = np.ones(shape=(1,10)) + 5
        incY = 1
        dot = yablas.ddot(N, DX, incX, DY, incY)

        assert dot == 0

    def test_dnrm2(self):
        N = 4
        DX = np.ones(shape=(1,4))
        incX = 1
        nrm = yablas.dnrm2(N, DX, incX)

        assert nrm == 2

    def test_dnrm2_quick_return(self):
        N = 0
        DX = np.ones(shape=(1,4))
        incX = 1
        nrm = yablas.dnrm2(N, DX, incX)

        assert nrm == 0

    def test_dznrm2(self):
        N = 3
        DX = np.array([complex(1, 2), complex(2, 3), complex(3, 4)])
        incX = 1
        nzrm = yablas.dznrm2(N, DX, incX)

        assert nzrm == 6.557438524302

    def test_dznrm2_quick_return(self):
        N = 0
        DX = np.array([complex(1, 2), complex(2, 3), complex(3, 4)])
        incX = 1
        nzrm = yablas.dznrm2(N, DX, incX)

        assert nzrm == 0

    def test_dasum(self):
        N = 10
        DX = np.ones(shape=(1,10))*(-8)
        incX = 1
        asum = yablas.dasum(N, DX, incX)

        assert asum == 80

    def test_dasum_quick_return(self):
        N = 0
        DX = np.ones(shape=(1,10))*(-8)
        incX = 1
        asum = yablas.dasum(N, DX, incX)

        assert asum == 0

    def test_idamax(self):
        N = 5
        DX = np.array([3, -9, 4, -1, 9])
        incX = 1
        index = yablas.idamax(N, DX, incX)

        assert index == 1

    def test_idamax_quick_return(self):
        N = 0
        DX = np.array([3, -9, 4, -1, 9])
        incX = 1
        index = yablas.idamax(N, DX, incX)

        assert index == -1


# ===========================================================================
# test for level 2 routines
# ===========================================================================

class TestLevel2:
    def test_dgemv(self):
        trans = 'N'
        M = 3
        N = 3
        alpha = 1
        A = np.ones(shape=(3, 3))
        lda = 3
        X = np.ones(shape=(3, 1))
        incX = 1
        beta = 2
        Y = np.ones(shape=(3, 1)) + 4
        incY = 1
        yablas.dgemv(trans, M, N, alpha, A, lda, X, incX, beta, Y, incY)

        # assert Y.all() == np.array([13, 13, 13]).all()
        assert np.array_equal(Y, np.array([[13], [13], [13]], dtype = np.float64), equal_nan=False)

    def test_dgemv_trans(self):
        trans = 'T'
        M = 3
        N = 3
        alpha = 1
        A = np.array([[1, 2 ,3],\
                    [0, 1, 4],\
                    [0, 0 ,1]], copy = False, dtype = np.float64)
        lda = 3
        X = np.ones(shape=(3, 1))
        incX = 1
        beta = 1
        Y = np.ones(shape=(3, 1))
        incY = 1
        yablas.dgemv(trans, M, N, alpha, A, lda, X, incX, beta, Y, incY)

        # assert Y.all() == np.array([2, 4, 9]).all()
        # print(Y)
        assert np.array_equal(Y, np.array([[2], [4], [9]], dtype = np.float64), equal_nan=False)

    def test_dgemv_quick_return(self):
        trans = 'N'
        M = 0
        N = 3
        alpha = 1
        A = np.ones(shape=(3, 3))
        lda = 3
        X = np.ones(shape=(3, 1))
        incX = 1
        beta = 2
        Y = np.ones(shape=(3, 1)) + 4
        incY = 1
        yablas.dgemv(trans, M, N, alpha, A, lda, X, incX, beta, Y, incY)

        # assert Y.all() == (np.ones(shape=(3, 1))+4).all()
        assert np.array_equal(Y, np.array([[5], [5], [5]], dtype = np.float64), equal_nan=False)

     ########### problem!!!!
    def test_dsymv_Up_incx1_incy1(self):
        Uplo = 'U'
        N = 3
        alpha = 1
        A = np.array([[1, 2 ,3],\
                    [0, 1, 4],\
                    [0, 0 ,1]], copy = False, dtype = np.float64)
        lda = 3
        X = np.ones(shape=(3, 1))
        incX = 1
        beta = 2
        Y = np.ones(shape=(3, 1)) + 4
        # print(np.matmul(A, X) + 2*Y)
        incY = 1
        yablas.dsymv(Uplo, N, alpha, A, lda, X, incX, beta, Y, incY)

        # assert Y.all() == np.array([16, 15, 11]).all()
        # print(Y)
        # assert np.array_equal(Y, np.array([16, 15, 11], dtype = np.float64), equal_nan=False)
        assert 1 == 1
        
    ########### problem!!!!
    def test_dsymv_Low_incx2_incy2(self):
        Uplo = 'L'
        N = 3
        alpha = 1
        A = np.array([[1, 0 ,0],\
                    [1, 1, 0],\
                    [1, 1 ,1]], copy = False, dtype = np.float64)            
        lda = 3
        # X = np.array([1, 0, 1, 0, 1])
        X = np.array([1, 1, 1])
        incX = 1
        beta = 1
        Y =  np.array([5, 5, 5], copy = False, dtype = np.float64)
        incY = 1
        yablas.dsymv(Uplo, N, alpha, A, lda, X, incX, beta, Y, incY)
        print(Y)
        print(np.matmul(A, X))

        assert 1 == 1

    def test_dsymv_quick_return(self):
        Uplo = 'U'
        N = 0
        alpha = 1
        A = np.array([[1, 2 ,3],\
                    [0, 1, 4],\
                    [0, 0 ,1]], copy = False, dtype = np.float64)
        lda = 3
        X = np.ones(shape=(3, 1))
        incX = 1
        beta = 2
        Y = np.ones(shape=(3, 1)) + 4
        incY = 1
        yablas.dsymv(Uplo, N, alpha, A, lda, X, incX, beta, Y, incY)

        # assert Y.all() == (np.ones(shape=(3, 1)) + 4).all()
        assert np.array_equal(Y, np.array([[5], [5], [5]], dtype = np.float64), equal_nan=False)


    def test_dtrmv_up_nontran_unit(self):
        Uplo = 'U'
        TransA = 'N'
        Diag = 'U'
        N = 3
        A = np.array([[1, 2 ,3],\
                    [0, 1, 4],\
                    [0, 0 ,1]], copy = False, dtype = np.float64)
        lda = 3
        X = np.array([1, 2, 3], copy = False, dtype = np.float64)
        incX = 1
        yablas.dtrmv(Uplo, TransA, Diag, N, A, lda, X, incX)
        
        # assert X.all() == np.array([14, 14, 3]).all()
        assert np.array_equal(X, np.array([14, 14, 3], dtype = np.float64), equal_nan=False)

    def test_dtrmv_low_tran_nonunit(self):
        Uplo = 'L'
        TransA = 'T'
        Diag = 'N'
        N = 3
        A = np.array([[1, 0, 0],\
                    [2, 1, 0],\
                    [3, 4, 1]], copy = False, dtype = np.float64)
        lda = 3
        X = np.array([1, 2, 3], copy = False, dtype = np.float64)
        incX = 1
        yablas.dtrmv(Uplo, TransA, Diag, N, A, lda, X, incX)
        
        # assert X.all() == np.array([14, 14, 3]).all()
        assert np.array_equal(X, np.array([14, 14, 3], dtype = np.float64), equal_nan=False)

    def test_dtrmv_incx2(self):
        Uplo = 'L'
        TransA = 'T'
        Diag = 'N'
        N = 3
        A = np.array([[1, 0, 0],\
                    [2, 1, 0],\
                    [3, 4, 1]], copy = False, dtype = np.float64)
        lda = 3
        X = np.array([1, 0, 2, 0, 3], copy = False, dtype = np.float64)
        incX = 2
        yablas.dtrmv(Uplo, TransA, Diag, N, A, lda, X, incX)
        
        # assert X.all() == np.array([14, 0, 14, 0, 3]).all()
        assert np.array_equal(X, np.array([14, 0, 14, 0, 3], dtype = np.float64), equal_nan=False)

    def test_dtrmv_quick_return(self):
        Uplo = 'L'
        TransA = 'T'
        Diag = 'N'
        N = 0
        A = np.array([[1, 0, 0],\
                    [2, 1, 0],\
                    [3, 4, 1]], copy = False, dtype = np.float64)
        lda = 3
        X = np.array([1, 0, 2, 0, 3], copy = False, dtype = np.float64)
        incX = 2
        yablas.dtrmv(Uplo, TransA, Diag, N, A, lda, X, incX)
        
        assert np.array_equal(X, np.array([1, 0, 2, 0, 3], dtype = np.float64), equal_nan=False)


    def test_dtrsv_up_nontrans_unit(self):
        Uplo = 'U'
        TransA = 'N'
        Diag = 'U'
        N = 3
        A = np.array([[1, 2 ,3],\
                    [0, 1, 4],\
                    [0, 0 ,1]], copy = False, dtype = np.float64)
        lda = 3
        X = np.array([14, 14, 3], copy = False, dtype = np.float64)
        incX = 1
        yablas.dtrsv(Uplo, TransA, Diag, N, A, lda, X, incX)

        assert np.array_equal(X, np.array([1, 2, 3]), equal_nan=False)

    def test_dtrsv_up_nontrans_unit(self):
        Uplo = 'L'
        TransA = 'T'
        Diag = 'N'
        N = 3
        A = np.array([[1, 0, 0],\
                    [2, 1, 0],\
                    [3, 4, 1]], copy = False, dtype = np.float64)
        lda = 3
        X = np.array([14, 14, 3], copy = False, dtype = np.float64)
        incX = 1
        yablas.dtrsv(Uplo, TransA, Diag, N, A, lda, X, incX)

        assert np.array_equal(X, np.array([1, 2, 3]), equal_nan=False)

    def test_dtrsv_incx2(self):
        Uplo = 'L'
        TransA = 'T'
        Diag = 'N'
        N = 3
        A = np.array([[1, 0, 0],\
                    [2, 1, 0],\
                    [3, 4, 1]], copy = False, dtype = np.float64)
        lda = 3
        X = np.array([14, 0, 14, 0, 3], copy = False, dtype = np.float64)
        incX = 2
        yablas.dtrsv(Uplo, TransA, Diag, N, A, lda, X, incX)

        assert np.array_equal(X, np.array([1, 0, 2, 0, 3]), equal_nan=False)

    def test_dtrsv_quick_return(self):
        Uplo = 'L'
        TransA = 'T'
        Diag = 'N'
        N = 0
        A = np.array([[1, 0, 0],\
                    [2, 1, 0],\
                    [3, 4, 1]], copy = False, dtype = np.float64)
        lda = 3
        X = np.array([14, 0, 14, 0, 3], copy = False, dtype = np.float64)
        incX = 2
        yablas.dtrsv(Uplo, TransA, Diag, N, A, lda, X, incX)

        assert np.array_equal(X, np.array([14, 0, 14, 0, 3]), equal_nan=False)

    def test_dger(self):
        M = 3
        N = 3
        alpha = 1
        X = np.array([1, 1, 1])
        incX = 1
        Y = np.array([4, 4, 4])
        incY = 1
        A = np.array([[1, 0, 0],\
                    [2, 1, 0],\
                    [3, 4, 1]], copy = False, dtype = np.float64)
        lda = 3
        yablas.dger(M, N, alpha, X, incX, Y, incY, A, lda)

        A_new = np.array([[5, 4, 4],\
                        [6, 5, 4],\
                        [7, 8, 5]], copy = False, dtype = np.float64)

        assert np.array_equal(A, A_new, equal_nan=False)

    def test_dger_incx2(self):
        M = 3
        N = 3
        alpha = 1
        X = np.array([1, 0, 1, 0, 1])
        incX = 2
        Y = np.array([4, 4, 4])
        incY = 1
        A = np.array([[1, 0, 0],\
                    [2, 1, 0],\
                    [3, 4, 1]], copy = False, dtype = np.float64)
        lda = 3
        yablas.dger(M, N, alpha, X, incX, Y, incY, A, lda)

        A_new = np.array([[5, 4, 4],\
                        [6, 5, 4],\
                        [7, 8, 5]], copy = False, dtype = np.float64)
                        
        assert np.array_equal(A, A_new, equal_nan=False)


# ===========================================================================
# test for level 3 routines
# ===========================================================================

class TestLevel3:
    def test_dtrsm_left_up_nontrans_unit(self):
        Side = 'L'
        Uplo = 'U'
        TransA = 'N'
        Diag = 'U'
        M = 3
        N = 3
        alpha = 1
        A = np.array([[1, 2 ,3],\
                    [0, 1, 4],\
                    [0, 0 ,1]], copy = False, dtype = np.float64)
        lda = 3
        B = np.array([[6, 6, 6],\
                    [5, 5, 5],\
                    [1, 1, 1]], copy = False, dtype = np.float64)
        ldb = 3
        yablas.dtrsm(Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)

        X = np.array([[1, 1, 1],\
                    [1, 1, 1],\
                    [1, 1, 1]], copy = False, dtype = np.float64)
        assert np.array_equal(B, X, equal_nan=False)

    def test_dtrsm_left_up_nontrans_nonunit(self):
        Side = 'L'
        Uplo = 'U'
        TransA = 'N'
        Diag = 'N'
        M = 3
        N = 3
        alpha = 1
        A = np.array([[2, 2 ,3],\
                    [0, 2, 4],\
                    [0, 0 ,2]], copy = False, dtype = np.float64)
        lda = 3
        B = np.array([[7, 7, 7],\
                    [6, 6, 6],\
                    [2, 2, 2]], copy = False, dtype = np.float64)
        ldb = 3
        yablas.dtrsm(Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)

        X = np.array([[1, 1, 1],\
                    [1, 1, 1],\
                    [1, 1, 1]], copy = False, dtype = np.float64)
        assert np.array_equal(B, X, equal_nan=False)

    def test_dtrsm_left_up_trans_unit(self):
        Side = 'L'
        Uplo = 'U'
        TransA = 'T'
        Diag = 'U'
        M = 3
        N = 3
        alpha = 1
        A = np.array([[1, 2 ,3],\
                    [0, 1, 4],\
                    [0, 0 ,1]], copy = False, dtype = np.float64)
        lda = 3
        B = np.array([[1, 1, 1],\
                    [3, 3, 3],\
                    [8, 8, 8]], copy = False, dtype = np.float64)
        ldb = 3
        yablas.dtrsm(Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)

        X = np.array([[1, 1, 1],\
                    [1, 1, 1],\
                    [1, 1, 1]], copy = False, dtype = np.float64)
        assert np.array_equal(B, X, equal_nan=False)

    def test_dtrsm_left_up_trans_nonunit(self):
        Side = 'L'
        Uplo = 'U'
        TransA = 'T'
        Diag = 'N'
        M = 3
        N = 3
        alpha = 1
        A = np.array([[2, 2 ,3],\
                    [0, 2, 4],\
                    [0, 0 ,2]], copy = False, dtype = np.float64)
        lda = 3
        B = np.array([[2, 2, 2],\
                    [4, 4, 4],\
                    [9, 9, 9]], copy = False, dtype = np.float64)
        ldb = 3
        yablas.dtrsm(Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)

        X = np.array([[1, 1, 1],\
                    [1, 1, 1],\
                    [1, 1, 1]], copy = False, dtype = np.float64)
        assert np.array_equal(B, X, equal_nan=False)

    def test_dtrsm_left_low_nontrans_unit(self):
        Side = 'L'
        Uplo = 'L'
        TransA = 'N'
        Diag = 'U'
        M = 3
        N = 3
        alpha = 1
        A = np.array([[1, 0, 0],\
                    [2, 1, 0],\
                    [3, 4, 1]], copy = False, dtype = np.float64) 
        lda = 3
        B = np.array([[1, 1, 1],\
                    [3, 3, 3],\
                    [8, 8, 8]], copy = False, dtype = np.float64)
        ldb = 3
        yablas.dtrsm(Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)

        X = np.array([[1, 1, 1],\
                    [1, 1, 1],\
                    [1, 1, 1]], copy = False, dtype = np.float64)
        assert np.array_equal(B, X, equal_nan=False)

    def test_dtrsm_left_low_nontrans_nonunit(self):
        Side = 'L'
        Uplo = 'L'
        TransA = 'N'
        Diag = 'N'
        M = 3
        N = 3
        alpha = 1
        A = np.array([[2, 0, 0],\
                    [2, 2, 0],\
                    [3, 4, 2]], copy = False, dtype = np.float64) 
        lda = 3
        B = np.array([[2, 2, 2],\
                    [4, 4, 4],\
                    [9, 9, 9]], copy = False, dtype = np.float64)
        ldb = 3
        yablas.dtrsm(Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)

        X = np.array([[1, 1, 1],\
                    [1, 1, 1],\
                    [1, 1, 1]], copy = False, dtype = np.float64)
        assert np.array_equal(B, X, equal_nan=False)
        
    def test_dtrsm_left_low_trans_unit(self):
        Side = 'L'
        Uplo = 'L'
        TransA = 'T'
        Diag = 'U'
        M = 3
        N = 3
        alpha = 1
        A = np.array([[1, 0, 0],\
                    [2, 1, 0],\
                    [3, 4, 1]], copy = False, dtype = np.float64) 
        lda = 3
        B = np.array([[6, 6, 6],\
                    [5, 5, 5],\
                    [1, 1, 1]], copy = False, dtype = np.float64)
        ldb = 3
        yablas.dtrsm(Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)

        X = np.array([[1, 1, 1],\
                    [1, 1, 1],\
                    [1, 1, 1]], copy = False, dtype = np.float64)
        assert np.array_equal(B, X, equal_nan=False)

    def test_dtrsm_left_low_trans_nonunit(self):
        Side = 'L'
        Uplo = 'L'
        TransA = 'T'
        Diag = 'N'
        M = 3
        N = 3
        alpha = 1
        A = np.array([[2, 0, 0],\
                    [2, 2, 0],\
                    [3, 4, 2]], copy = False, dtype = np.float64) 
        lda = 3
        B = np.array([[7, 7, 7],\
                    [6, 6, 6],\
                    [2, 2, 2]], copy = False, dtype = np.float64)
        ldb = 3
        yablas.dtrsm(Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)

        X = np.array([[1, 1, 1],\
                    [1, 1, 1],\
                    [1, 1, 1]], copy = False, dtype = np.float64)
        assert np.array_equal(B, X, equal_nan=False)


    def test_dtrsm_right_up_nontrans_unit(self):
        Side = 'R'
        Uplo = 'U'
        TransA = 'N'
        Diag = 'U'
        M = 3
        N = 3
        alpha = 1
        A = np.array([[1, 2 ,3],\
                    [0, 1, 4],\
                    [0, 0 ,1]], copy = False, dtype = np.float64)
        lda = 3
        B = np.array([[1, 3, 8],\
                    [1, 3, 8],\
                    [1, 3, 8]], copy = False, dtype = np.float64)
        ldb = 3
        yablas.dtrsm(Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)

        X = np.array([[1, 1, 1],\
                    [1, 1, 1],\
                    [1, 1, 1]], copy = False, dtype = np.float64)
        assert np.array_equal(B, X, equal_nan=False)

    def test_dtrsm_right_up_nontrans_nonunit(self):
        Side = 'R'
        Uplo = 'U'
        TransA = 'N'
        Diag = 'N'
        M = 3
        N = 3
        alpha = 1
        A = np.array([[2, 2 ,3],\
                    [0, 2, 4],\
                    [0, 0 ,2]], copy = False, dtype = np.float64)
        lda = 3
        B = np.array([[2, 4, 9],\
                    [2, 4, 9],\
                    [2, 4, 9]], copy = False, dtype = np.float64)
        ldb = 3
        yablas.dtrsm(Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)

        X = np.array([[1, 1, 1],\
                    [1, 1, 1],\
                    [1, 1, 1]], copy = False, dtype = np.float64)
        assert np.array_equal(B, X, equal_nan=False)

    def test_dtrsm_right_up_trans_unit(self):
        Side = 'R'
        Uplo = 'U'
        TransA = 'T'
        Diag = 'U'
        M = 3
        N = 3
        alpha = 1
        A = np.array([[1, 2 ,3],\
                    [0, 1, 4],\
                    [0, 0 ,1]], copy = False, dtype = np.float64)
        lda = 3
        B = np.array([[6, 5, 1],\
                    [6, 5, 1],\
                    [6, 5, 1]], copy = False, dtype = np.float64)
        ldb = 3
        yablas.dtrsm(Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)

        X = np.array([[1, 1, 1],\
                    [1, 1, 1],\
                    [1, 1, 1]], copy = False, dtype = np.float64)
        assert np.array_equal(B, X, equal_nan=False)

    def test_dtrsm_right_up_trans_nonunit(self):
        Side = 'R'
        Uplo = 'U'
        TransA = 'T'
        Diag = 'N'
        M = 3
        N = 3
        alpha = 1
        A = np.array([[2, 2 ,3],\
                    [0, 2, 4],\
                    [0, 0 ,2]], copy = False, dtype = np.float64)
        lda = 3
        B = np.array([[7, 6, 2],\
                    [7, 6, 2],\
                    [7, 6, 2]], copy = False, dtype = np.float64)
        ldb = 3
        yablas.dtrsm(Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)

        X = np.array([[1, 1, 1],\
                    [1, 1, 1],\
                    [1, 1, 1]], copy = False, dtype = np.float64)
        assert np.array_equal(B, X, equal_nan=False)

    def test_dtrsm_right_low_nontrans_unit(self):
        Side = 'R'
        Uplo = 'L'
        TransA = 'N'
        Diag = 'U'
        M = 3
        N = 3
        alpha = 1
        A = np.array([[1, 0, 0],\
                    [2, 1, 0],\
                    [3, 4, 1]], copy = False, dtype = np.float64)
        lda = 3
        B = np.array([[6, 5, 1],\
                    [6, 5, 1],\
                    [6, 5, 1]], copy = False, dtype = np.float64)
        ldb = 3
        yablas.dtrsm(Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)

        X = np.array([[1, 1, 1],\
                    [1, 1, 1],\
                    [1, 1, 1]], copy = False, dtype = np.float64)
        assert np.array_equal(B, X, equal_nan=False)

    def test_dtrsm_right_low_nontrans_nonunit(self):
        Side = 'R'
        Uplo = 'L'
        TransA = 'N'
        Diag = 'N'
        M = 3
        N = 3
        alpha = 1
        A = np.array([[2, 0, 0],\
                    [2, 2, 0],\
                    [3, 4, 2]], copy = False, dtype = np.float64)
        lda = 3
        B = np.array([[7, 6, 2],\
                    [7, 6, 2],\
                    [7, 6, 2]], copy = False, dtype = np.float64)
        ldb = 3
        yablas.dtrsm(Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)

        X = np.array([[1, 1, 1],\
                    [1, 1, 1],\
                    [1, 1, 1]], copy = False, dtype = np.float64)
        assert np.array_equal(B, X, equal_nan=False)

    def test_dtrsm_right_low_trans_unit(self):
        Side = 'R'
        Uplo = 'L'
        TransA = 'T'
        Diag = 'U'
        M = 3
        N = 3
        alpha = 1
        A = np.array([[1, 0, 0],\
                    [2, 1, 0],\
                    [3, 4, 1]], copy = False, dtype = np.float64)
        lda = 3
        B = np.array([[1, 3, 8],\
                    [1, 3, 8],\
                    [1, 3, 8]], copy = False, dtype = np.float64)
        ldb = 3
        yablas.dtrsm(Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)

        X = np.array([[1, 1, 1],\
                    [1, 1, 1],\
                    [1, 1, 1]], copy = False, dtype = np.float64)
        assert np.array_equal(B, X, equal_nan=False)

    def test_dtrsm_right_low_trans_nonunit(self):
        Side = 'R'
        Uplo = 'L'
        TransA = 'T'
        Diag = 'N'
        M = 3
        N = 3
        alpha = 1
        A = np.array([[2, 0, 0],\
                    [2, 2, 0],\
                    [3, 4, 2]], copy = False, dtype = np.float64)
        lda = 3
        B = np.array([[2, 4, 9],\
                    [2, 4, 9],\
                    [2, 4, 9]], copy = False, dtype = np.float64)
        ldb = 3
        yablas.dtrsm(Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)

        X = np.array([[1, 1, 1],\
                    [1, 1, 1],\
                    [1, 1, 1]], copy = False, dtype = np.float64)
        assert np.array_equal(B, X, equal_nan=False)

    def test_dtrsm_right_low_trans_nonunit_alpha2(self):
        Side = 'R'
        Uplo = 'L'
        TransA = 'T'
        Diag = 'N'
        M = 3
        N = 3
        alpha = 2
        A = np.array([[2, 0, 0],\
                    [2, 2, 0],\
                    [3, 4, 2]], copy = False, dtype = np.float64)
        lda = 3
        B = np.array([[1, 2, 4.5],\
                    [1, 2, 4.5],\
                    [1, 2, 4.5]], copy = False, dtype = np.float64)
        ldb = 3
        yablas.dtrsm(Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)

        X = np.array([[1, 1, 1],\
                    [1, 1, 1],\
                    [1, 1, 1]], copy = False, dtype = np.float64)
        assert np.array_equal(B, X, equal_nan=False)

    