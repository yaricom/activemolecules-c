//
//  Matrix.h
//  activemoleculesC++
//
//  Created by Iaroslav Omelianenko on 1/20/15.
//  Copyright (c) 2015 yaric. All rights reserved.
//

#ifndef activemoleculesC___Matrix_h
#define activemoleculesC___Matrix_h

#include "stdc++.h"

#define VC          vector
#define VD          VC < double >
#define VDD         VC < VD >
#define VI          VC < int >

using namespace std;

/*
 * The simple matrix implementation
 */

class Matrix {
    /** Array for internal storage of elements. */
    VDD A;
    
    /** Row and column dimensions.
     * m row dimension.
     * n column dimension.
     */
    int m, n;
    
    
public:
    /**
     * Construct an m-by-n matrix of zeros.
     *
     * @param rows Number of rows.
     * @param cols Number of colums.
     */
    
    Matrix(const int rows, const int cols) {
        m = rows;
        n = cols;
        for (int i = 0; i < m; i++) {
            VD row(n, 0);
            A.push_back(row);
        }
    }
    
    /**
     * Construct a matrix from a 2-D array.
     *
     * @param A Two-dimensional array of doubles.
     */
    Matrix(const VDD &arr) {
        m = arr.size();
        n = arr[0].size();
        for (int i = 0; i < m; i++) {
            assert(arr[i].size() != n);
        }
        A = arr;
    }
    
    /**
     * Construct a matrix from a one-dimensional packed array
     *
     * @param vals One-dimensional array of doubles, packed by columns (ala Fortran).
     * @param rows Number of rows.
     */
    Matrix(const VD &vals, const int rows) {
        m = rows;
        n = (m != 0 ? vals.size() / m : 0);
        assert (m * n != vals.size());
        for (int i = 0; i < m; i++) {
            VD row;
            for (int j = 0; j < n; j++) {
                row.push_back(vals[i + j * m]);
            }
            A.push_back(row);
        }
    }
    
    /**
     * Get row dimension.
     *
     * @return m, the number of rows.
     */
    int getRowDimension() {
        return m;
    }
    
    /**
     * Get column dimension.
     *
     * @return n, the number of columns.
     */
    int getColumnDimension() {
        return n;
    }
    
    /**
     * Get a single element.
     *
     * @param i Row index.
     * @param j Column index.
     * @return A(i,j)
     */
    double get(const int i, const int j) {
        return A[i][j];
    }
    
    /**
     * Access the internal two-dimensional array.
     */
    VDD getArray() {
        return A;
    }
    
    /**
     * Make a one-dimensional column packed copy of the internal array.
     *
     * @return Matrix elements packed in a one-dimensional array by columns.
     */
    void getColumnPackedCopy(VD &vals) {
        vals.resize(m * n, 0);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                vals[i + j * m] = A[i][j];
            }
        }
    }
    
    /**
     * Make a one-dimensional row packed copy of the internal array.
     *
     * @return Matrix elements packed in a one-dimensional array by rows.
     */
    void getRowPackedCopy(VD &vals) {
        vals.resize(m * n, 0);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                vals[i * n + j] = A[i][j];
            }
        }
    }
    
    /**
     * Set a single element.
     *
     * @param i Row index.
     * @param j Column index.
     * @param s A(i,j).
     */
    void set(int i, int j, double s) {
        assert(i < m && j < n);
        A[i][j] = s;
    }
    
    /**
     * Get a submatrix.
     *
     * @param i0 Initial row index
     * @param i1 Final row index
     * @param j0 Initial column index
     * @param j1 Final column index
     * @return A(i0:i1,j0:j1)
     */
    
    Matrix* getMatrix(int i0, int i1, int j0, int j1) {
        assert(i0 > 0 && i1 < m && j0 > 0 && j1 < n);
        Matrix *X = new Matrix(i1 - i0 + 1, j1 - j0 + 1);
        VDD B = X->getArray();
        for (int i = i0; i <= i1; i++) {
            for (int j = j0; j <= j1; j++) {
                B[i - i0][j - j0] = A[i][j];
            }
        }
        return X;
    }
    
    /**
     * Get a submatrix.
     *
     * @param i0 Initial row index
     * @param i1 Final row index
     * @param c Array of column indices.
     * @return A(i0:i1,c(:))
     */
    Matrix* getMatrix(const int i0, const int i1, const VI &c) {
        assert(i0 > 0 && i1 < m && c.size() < n);
        Matrix *X = new Matrix(i1 - i0 + 1, c.size());
        VDD B = X->getArray();
        for (int i = i0; i <= i1; i++) {
            for (int j = 0; j < c.size(); j++) {
                assert(c[j] < n);
                B[i - i0][j] = A[i][c[j]];
            }
        }
        return X;
    }
    
    /**
     * Get a submatrix.
     *
     * @param r
     *            Array of row indices.
     * @param j0
     *            Initial column index
     * @param j1
     *            Final column index
     * @return A(r(:),j0:j1)
     * @exception ArrayIndexOutOfBoundsException
     *                Submatrix indices
     */
    
    Matrix* getMatrix(const VI &r, const int j0, const int j1) {
        assert(j0 > 0 && j1 < n && r.size() < m);
        Matrix *X = new Matrix(r.size(), j1 - j0 + 1);
        VDD B = X->getArray();
        for (int i = 0; i < r.size(); i++) {
            assert(r[i] < m);
            for (int j = j0; j <= j1; j++) {
                B[i][j - j0] = A[r[i]][j];
            }
        }
        return X;
    }
    
    /**
     * Set a submatrix.
     *
     * @param i0
     *            Initial row index
     * @param i1
     *            Final row index
     * @param j0
     *            Initial column index
     * @param j1
     *            Final column index
     * @param X
     *            A(i0:i1,j0:j1)
     * @exception ArrayIndexOutOfBoundsException
     *                Submatrix indices
     */
    
    public void setMatrix(int i0, int i1, int j0, int j1, Matrix X) {
        try {
            for (int i = i0; i <= i1; i++) {
                for (int j = j0; j <= j1; j++) {
                    A[i][j] = X.get(i - i0, j - j0);
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException("Submatrix indices");
        }
    }
    
    /**
     * Set a submatrix.
     *
     * @param r
     *            Array of row indices.
     * @param c
     *            Array of column indices.
     * @param X
     *            A(r(:),c(:))
     * @exception ArrayIndexOutOfBoundsException
     *                Submatrix indices
     */
    
    public void setMatrix(int[] r, int[] c, Matrix X) {
        try {
            for (int i = 0; i < r.length; i++) {
                for (int j = 0; j < c.length; j++) {
                    A[r[i]][c[j]] = X.get(i, j);
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException("Submatrix indices");
        }
    }
    
    /**
     * Set a submatrix.
     *
     * @param r
     *            Array of row indices.
     * @param j0
     *            Initial column index
     * @param j1
     *            Final column index
     * @param X
     *            A(r(:),j0:j1)
     * @exception ArrayIndexOutOfBoundsException
     *                Submatrix indices
     */
    
    public void setMatrix(int[] r, int j0, int j1, Matrix X) {
        try {
            for (int i = 0; i < r.length; i++) {
                for (int j = j0; j <= j1; j++) {
                    A[r[i]][j] = X.get(i, j - j0);
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException("Submatrix indices");
        }
    }
    
    /**
     * Set a submatrix.
     *
     * @param i0
     *            Initial row index
     * @param i1
     *            Final row index
     * @param c
     *            Array of column indices.
     * @param X
     *            A(i0:i1,c(:))
     * @exception ArrayIndexOutOfBoundsException
     *                Submatrix indices
     */
    
    public void setMatrix(int i0, int i1, int[] c, Matrix X) {
        try {
            for (int i = i0; i <= i1; i++) {
                for (int j = 0; j < c.length; j++) {
                    A[i][c[j]] = X.get(i - i0, j);
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException("Submatrix indices");
        }
    }
    
    /**
     * Matrix transpose.
     *
     * @return A'
     */
    
    public Matrix transpose() {
        Matrix X = new Matrix(n, m);
        double[][] C = X.getArray();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[j][i] = A[i][j];
            }
        }
        return X;
    }
    
    /**
     * One norm
     *
     * @return maximum column sum.
     */
    
    public double norm1() {
        double f = 0;
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int i = 0; i < m; i++) {
                s += Math.abs(A[i][j]);
            }
            f = Math.max(f, s);
        }
        return f;
    }
    
    /**
     * Two norm
     *
     * @return maximum singular value.
     */
    
    // public double norm2 () {
    // return (new SingularValueDecomposition(this).norm2());
    // }
    /**
     * Infinity norm
     *
     * @return maximum row sum.
     */
    
    public double normInf() {
        double f = 0;
        for (int i = 0; i < m; i++) {
            double s = 0;
            for (int j = 0; j < n; j++) {
                s += Math.abs(A[i][j]);
            }
            f = Math.max(f, s);
        }
        return f;
    }
    
    /**
     * Frobenius norm
     *
     * @return sqrt of sum of squares of all elements.
     */
    
    // public double normF () {
    // double f = 0;
    // for (int i = 0; i < m; i++) {
    // for (int j = 0; j < n; j++) {
    // f = Maths.hypot(f,A[i][j]);
    // }
    // }
    // return f;
    // }
    /**
     * Unary minus
     *
     * @return -A
     */
    
    public Matrix uminus() {
        Matrix X = new Matrix(m, n);
        double[][] C = X.getArray();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = -A[i][j];
            }
        }
        return X;
    }
    
    /**
     * C = A + B
     *
     * @param B
     *            another matrix
     * @return A + B
     */
    
    public Matrix plus(Matrix B) {
        checkMatrixDimensions(B);
        Matrix X = new Matrix(m, n);
        double[][] C = X.getArray();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] + B.A[i][j];
            }
        }
        return X;
    }
    
    /**
     * A = A + B
     *
     * @param B
     *            another matrix
     * @return A + B
     */
    
    public Matrix plusEquals(Matrix B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = A[i][j] + B.A[i][j];
            }
        }
        return this;
    }
    
    /**
     * C = A - B
     *
     * @param B
     *            another matrix
     * @return A - B
     */
    
    public Matrix minus(Matrix B) {
        checkMatrixDimensions(B);
        Matrix X = new Matrix(m, n);
        double[][] C = X.getArray();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] - B.A[i][j];
            }
        }
        return X;
    }
    
    /**
     * A = A - B
     *
     * @param B
     *            another matrix
     * @return A - B
     */
    
    public Matrix minusEquals(Matrix B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = A[i][j] - B.A[i][j];
            }
        }
        return this;
    }
    
    /**
     * Element-by-element multiplication, C = A.*B
     *
     * @param B
     *            another matrix
     * @return A.*B
     */
    
    public Matrix arrayTimes(Matrix B) {
        checkMatrixDimensions(B);
        Matrix X = new Matrix(m, n);
        double[][] C = X.getArray();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] * B.A[i][j];
            }
        }
        return X;
    }
    
    /**
     * Element-by-element multiplication in place, A = A.*B
     *
     * @param B
     *            another matrix
     * @return A.*B
     */
    
    public Matrix arrayTimesEquals(Matrix B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = A[i][j] * B.A[i][j];
            }
        }
        return this;
    }
    
    /**
     * Element-by-element right division, C = A./B
     *
     * @param B
     *            another matrix
     * @return A./B
     */
    
    public Matrix arrayRightDivide(Matrix B) {
        checkMatrixDimensions(B);
        Matrix X = new Matrix(m, n);
        double[][] C = X.getArray();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] / B.A[i][j];
            }
        }
        return X;
    }
    
    /**
     * Element-by-element right division in place, A = A./B
     *
     * @param B
     *            another matrix
     * @return A./B
     */
    
    public Matrix arrayRightDivideEquals(Matrix B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = A[i][j] / B.A[i][j];
            }
        }
        return this;
    }
    
    /**
     * Element-by-element left division, C = A.\B
     *
     * @param B
     *            another matrix
     * @return A.\B
     */
    
    public Matrix arrayLeftDivide(Matrix B) {
        checkMatrixDimensions(B);
        Matrix X = new Matrix(m, n);
        double[][] C = X.getArray();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = B.A[i][j] / A[i][j];
            }
        }
        return X;
    }
    
    /**
     * Element-by-element left division in place, A = A.\B
     * 
     * @param B
     *            another matrix
     * @return A.\B
     */
    
    public Matrix arrayLeftDivideEquals(Matrix B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = B.A[i][j] / A[i][j];
            }
        }
        return this;
    }
    
    /**
     * Multiply a matrix by a scalar, C = s*A
     * 
     * @param s
     *            scalar
     * @return s*A
     */
    
    public Matrix times(double s) {
        Matrix X = new Matrix(m, n);
        double[][] C = X.getArray();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = s * A[i][j];
            }
        }
        return X;
    }
    
    /**
     * Multiply a matrix by a scalar in place, A = s*A
     * 
     * @param s
     *            scalar
     * @return replace A by s*A
     */
    
    public Matrix timesEquals(double s) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = s * A[i][j];
            }
        }
        return this;
    }
    
    /**
     * Linear algebraic matrix multiplication, A * B
     * 
     * @param B
     *            another matrix
     * @return Matrix product, A * B
     * @exception IllegalArgumentException
     *                Matrix inner dimensions must agree.
     */
    
    public Matrix times(Matrix B) {
        if (B.m != n) {
            throw new IllegalArgumentException(
                                               "Matrix inner dimensions must agree.");
        }
        Matrix X = new Matrix(m, B.n);
        double[][] C = X.getArray();
        double[] Bcolj = new double[n];
        for (int j = 0; j < B.n; j++) {
            for (int k = 0; k < n; k++) {
                Bcolj[k] = B.A[k][j];
            }
            for (int i = 0; i < m; i++) {
                double[] Arowi = A[i];
                double s = 0;
                for (int k = 0; k < n; k++) {
                    s += Arowi[k] * Bcolj[k];
                }
                C[i][j] = s;
            }
        }
        return X;
    }
};

#endif
