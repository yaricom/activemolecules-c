//
//  Matrix.h
//  activemoleculesC++
//
//  Created by Iaroslav Omelianenko on 1/20/15.
//  Copyright (c) 2015 yaric. All rights reserved.
//

#ifndef activemoleculesC___Matrix_h
#define activemoleculesC___Matrix_h

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
     * Adds row of data
     *
     * @param vals the row of data to add
     */
    void addRow(const VD &vals) {
        assert(vals.size() == n);
        A.push_back(vals);
    }
    
    /**
     * Get row dimension.
     *
     * @return m, the number of rows.
     */
    int getRowDimension() const {
        return m;
    }
    
    /**
     * Get column dimension.
     *
     * @return n, the number of columns.
     */
    int getColumnDimension() const {
        return n;
    }
    
    /**
     * Get a single element.
     *
     * @param i Row index.
     * @param j Column index.
     * @return A(i,j)
     */
    double get(const int i, const int j) const {
        return A[i][j];
    }
    
    /**
     * Access the internal two-dimensional array.
     */
    VDD* getArray() {
        return &A;
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
    inline void set(int i, int j, double s) {
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
    
    Matrix getMatrix(int i0, int i1, int j0, int j1) {
        assert(i0 >= 0 && i0 < i1 && i1 < m && j0 >= 0 && j0 < j1 && j1 < n);
        Matrix X(i1 - i0 + 1, j1 - j0 + 1);
        for (int i = i0; i <= i1; i++) {
            for (int j = j0; j <= j1; j++) {
                X.set(i - i0, j - j0, A[i][j]);
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
    Matrix getMatrix(const int i0, const int i1, const VI &c) {
        assert(i0 >= 0 && i0 < i1 && i1 < m);
        Matrix X(i1 - i0 + 1, c.size());
        for (int i = i0; i <= i1; i++) {
            for (int j = 0; j < c.size(); j++) {
                assert(c[j] < n && c[j] >= 0);
                X.set(i - i0, j, A[i][c[j]]);
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
    
    Matrix getMatrix(const VI &r, const int j0, const int j1) {
        assert(j0 >= 0 && j0 < j1 && j1 < n);
        Matrix X(r.size(), j1 - j0 + 1);
        for (int i = 0; i < r.size(); i++) {
            assert(r[i] < m && r[i] >= 0);
            for (int j = j0; j <= j1; j++) {
                X.set(i, j - j0, A[r[i]][j]);
            }
        }
        return X;
    }
    
    /**
     * Set a submatrix.
     *
     * @param i0 Initial row index
     * @param i1 Final row index
     * @param j0 Initial column index
     * @param j1 Final column index
     * @param X A(i0:i1,j0:j1)
     */
    void setMatrix(const int i0, const int i1, const int j0, const int j1, const Matrix &X) {
        assert(i0 >= 0 && i0 < i1 && i1 < m && j0 >= 0 && j0 < j1 && j1 < n);
        for (int i = i0; i <= i1; i++) {
            for (int j = j0; j <= j1; j++) {
                A[i][j] = X.get(i - i0, j - j0);
            }
        }
    }
    
    /**
     * Set a submatrix.
     *
     * @param r Array of row indices.
     * @param c Array of column indices.
     * @param X A(r(:),c(:))
     */
    void setMatrix(const VI &r, const VI &c, const Matrix &X) {
        for (int i = 0; i < r.size(); i++) {
            assert(r[i] < m && r[i] >= 0);
            for (int j = 0; j < c.size(); j++) {
                assert(c[j] < n && c[j] >= 0);
                A[r[i]][c[j]] = X.get(i, j);
            }
        }
    }
    
    /**
     * Set a submatrix.
     *
     * @param r Array of row indices.
     * @param j0 Initial column index
     * @param j1 Final column index
     * @param X A(r(:),j0:j1)
     */
    void setMatrix(const VI &r, const int j0, const int j1, const Matrix &X) {
        assert(j0 >=0 && j0 < j1 && j1 < n);
        for (int i = 0; i < r.size(); i++) {
            assert(r[i] < m && r[i] >= 0);
            for (int j = j0; j <= j1; j++) {
                A[r[i]][j] = X.get(i, j - j0);
            }
        }
    }
    
    /**
     * Set a submatrix.
     *
     * @param i0 Initial row index
     * @param i1 Final row index
     * @param c Array of column indices.
     * @param X A(i0:i1,c(:))
     */
    void setMatrix(const int i0, const int i1, const VI c, const Matrix &X) {
        assert(i0 >= 0 && i0 < i1 && i1 < m);
        for (int i = i0; i <= i1; i++) {
            for (int j = 0; j < c.size(); j++) {
                assert(c[j] < n && c[j] >= 0);
                A[i][c[j]] = X.get(i - i0, j);
            }
        }
    }
    
    /**
     * Matrix transpose.
     *
     * @return A'
     */
    Matrix transpose() {
        Matrix X(n, m);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X.set(j, i, A[i][j]);
            }
        }
        return X;
    }
    
    /**
     * One norm
     *
     * @return maximum column sum.
     */
    double norm1() {
        double f = 0;
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int i = 0; i < m; i++) {
                s += abs(A[i][j]);
            }
            f = max(f, s);
        }
        return f;
    }
    
    /**
     * Infinity norm
     *
     * @return maximum row sum.
     */
    double normInf() {
        double f = 0;
        for (int i = 0; i < m; i++) {
            double s = 0;
            for (int j = 0; j < n; j++) {
                s += abs(A[i][j]);
            }
            f = max(f, s);
        }
        return f;
    }
    
    /**
     * Frobenius norm
     *
     * @return sqrt of sum of squares of all elements.
     */
    double normF () {
        double f = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                f = hypot(f, A[i][j]);
            }
        }
        return f;
    }
    /**
     * Unary minus
     *
     * @return -A
     */
    Matrix uminus() {
        Matrix X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X.set(i, j, -A[i][j]);
            }
        }
        return X;
    }
    
    /**
     * C = A + B
     *
     * @param B another matrix
     * @return A + B
     */
    Matrix operator+(const Matrix& B) {
        checkMatrixDimensions(B);
        Matrix X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X.set(i, j, this->A[i][j] + B.A[i][j]);
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
    
    Matrix& operator+=(const Matrix &B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->set(i, j, this->A[i][j] + B.A[i][j]);
            }
        }
        return *this;
    }
    
    /**
     * C = A - B
     *
     * @param B another matrix
     * @return A - B
     */
    
    Matrix operator-(const Matrix &B) {
        checkMatrixDimensions(B);
        Matrix X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X.set(i, j, this->A[i][j] - B.A[i][j]);
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
    
    Matrix& operator-=(const Matrix &B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->set(i, j, this->A[i][j] - B.A[i][j]);
            }
        }
        return *this;
    }
    
    /**
     * Element-by-element multiplication, C = A.*B
     * the Hadamard product (also known as the Schur product [1] or the entrywise product[2])
     *
     * @param B another matrix
     * @return A.*B
     */
    Matrix operator*(const Matrix &B) {
        checkMatrixDimensions(B);
        Matrix X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X.set(i, j, this->A[i][j] * B.A[i][j]);
            }
        }
        return X;
    }
    
    /**
     * Element-by-element multiplication in place, A = A.*B
     * the Hadamard product (also known as the Schur product [1] or the entrywise product[2])
     *
     * @param B another matrix
     * @return A.*B
     */
    
    Matrix& operator*=(const Matrix &B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->set(i, j, this->A[i][j] * B.A[i][j]);
            }
        }
        return *this;
    }
    
    /**
     * Element-by-element right division, C = A./B
     *
     * @param B
     *            another matrix
     * @return A./B
     */
    
    Matrix operator/(const Matrix &B) {
        checkMatrixDimensions(B);
        Matrix X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X.set(i, j, this->A[i][j] / B.A[i][j]);
            }
        }
        return X;
    }
    
    /**
     * Element-by-element right division in place, A = A./B
     *
     * @param B another matrix
     * @return A./B
     */
    
    Matrix& operator/=(const Matrix &B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->set(i, j, this->A[i][j] / B.A[i][j]);
            }
        }
        return *this;
    }
    
    /**
     * Element-by-element left division, C = A.\B
     *
     * @param B another matrix
     * @return A.\B
     */
    
    Matrix arrayLeftDivide(const Matrix &B) {
        checkMatrixDimensions(B);
        Matrix X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X.set(i, j, B.A[i][j] / this->A[i][j]);
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
    
    Matrix& arrayLeftDivideEquals(const Matrix &B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->set(i, j, B.A[i][j] / this->A[i][j]);
            }
        }
        return *this;
    }
    
    /**
     * Multiply a matrix by a scalar, C = s*A
     * 
     * @param s scalar
     * @return s*A
     */
    
    Matrix operator*(const double s) {
        Matrix X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X.set(i, j, s * this->A[i][j]);
            }
        }
        return X;
    }
    
    /**
     * Multiply a matrix by a scalar in place, A = s*A
     * 
     * @param s scalar
     * @return replace A by s*A
     */
    
    Matrix& operator*=(const double s) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->set(i, j, s * this->A[i][j]);
            }
        }
        return *this;
    }
    
    /**
     * Linear algebraic matrix multiplication, A * B
     * Matrix product
     * 
     * @param B another matrix
     * @return Matrix product, A x B
     */
    
    Matrix matmul(const Matrix &B) {
        // Matrix inner dimensions must agree."
        assert (B.m != n);
        
        Matrix X(m, B.n);
        double Bcolj[n];
        for (int j = 0; j < B.n; j++) {
            for (int k = 0; k < n; k++) {
                Bcolj[k] = B.A[k][j];
            }
            for (int i = 0; i < m; i++) {
                VD Arowi = A[i];
                double s = 0;
                for (int k = 0; k < n; k++) {
                    s += Arowi[k] * Bcolj[k];
                }
                X.set(i, j, s);
            }
        }
        return X;
    }

	/**
     * Calculates matrix mean by columns
     */
    Matrix mean() {
        Matrix current(1, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                current.A[0][n] += this->A[i][n];
            }
        }
        current /= m;
        return current;
    }
    
private:
    void checkMatrixDimensions(Matrix B) {
        assert (B.m != m || B.n != n);
    }
};

#endif
