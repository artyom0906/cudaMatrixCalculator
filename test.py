import numpy as np
import pprint
import scipy
import scipy.linalg  # SciPy Linear Algebra Library


def backward_substitution(U, b):
    m = len(b)
    x = np.empty(m)

    for v in range(m - 1, -1, -1):
        if U[v][v] == 0:
            # the value on the diagonal is zero
            x[v] = 0
            continue
        # calculate v-th variable value
        value = b[v]
        # subtract linear combination of the already known variables
        # in the top right corner of the matrix
        for i in range(v + 1, m, 1):
            value -= U[v][i] * x[i]
        # divide by the coefficient before the i-th variable to get it's value
        value /= U[v][v]
        # store the value in the resulting vector
        x[v] = value

    return x


# Solves a linear system of equations Lx = b where the L matrix is lower-triangular
def forward_substitution(L, b):
    m = len(b)
    x = np.empty(m)

    for v in range(m):
        if L[v][v] == 0:
            # the value on the diagonal is zero
            x[v] = 0
            # the current variable's value is irrelevant
            continue
        # calculate v-th variable value
        value = b[v]
        # subtract linear combination of the already known variables
        # in the bottom left corner of the matrix
        for i in range(v):
            value -= L[v][i] * x[i]
        print(value)
        # divide by the coefficient by the v-th variable to get it's value
        value /= L[v][v]
        # store the value in the resulting vector
        x[v] = value

    return x


def mult_matrix(M, N):
    """Multiply square matrices of same dimension M and N"""

    # Converts N into a list of tuples of columns
    tuple_N = zip(*N)

    # Nested list comprehension to calculate matrix multiplication
    return [[sum(el_m * el_n for el_m, el_n in zip(row_m, col_n)) for col_n in tuple_N] for row_m in M]


# Compute the PA = LU decomposition of the matrix A
def plu(A):
    m = len(A)

    P = np.identity(m)  # create an identity matrix of size m
    L = np.identity(m)

    for x in range(m):

        pivotRow = x

        if A[pivotRow][x] == 0:
            # search for a non-zero pivot
            for y in range(x + 1, m, 1):
                if A[y][x] != 0:
                    pivotRow = y
                    break

        if A[pivotRow][x] == 0:
            # we didn't find any row with a non-zero leading coefficient
            # that means that the matrix has all zeroes in this column
            # so we don't need to search for pivots after all for the current column x
            continue

        # did we just use a pivot that is not on the diagonal?
        if pivotRow != x:
            # so we need to swap the part of the pivot row after x including x
            # with the same right part of the x row where the pivot was expected
            for i in range(x, m, 1):
                # swap the two values columnwise
                (A[x][i], A[pivotRow][i]) = (A[pivotRow][i], A[x][i])

            # we must save the fact that we did this swap in the permutation matrix
            # swap the pivot row with row x
            P[[x, pivotRow]] = P[[pivotRow, x]]

            # we also need to swap the rows in the L matrix
            # however, the relevant part of the L matrix is only the bottom-left corner
            # and before x
            for i in range(x):
                (L[x][i], L[pivotRow][i]) = (L[pivotRow][i], L[x][i])

        # now the pivot row is x
        # search for rows where the leading coefficient must be eliminated
        for y in range(x + 1, m, 1):
            currentValue = A[y][x]
            if currentValue == 0:
                # variable already eliminated, nothing to do
                continue

            pivot = A[x][x]
            assert pivot != 0  # just in case, we already made sure the pivot is not zero

            pivotFactor = currentValue / pivot

            # subtract the pivot row from the current row
            A[y][x] = 0
            for i in range(x + 1, m, 1):
                A[y][i] -= pivotFactor * A[x][i]

            # write the pivot factor to the L matrix
            L[y][x] = pivotFactor

        # the pivot must anyway be at the correct position if we found at least one
        # non-zero leading coefficient in the current column x
        assert A[x][x] != 0
        for y in range(x + 1, m, 1):
            assert A[y][x] == 0

    return (P, L, A)


def plu_solve():
    A = [[2., 7., -5., 2., ],
         [2., -3., 0., -10., ],
         [-8., -10., -4., 1., ],
         [-6., -3., -8., 4.]
         ]
    P = np.array([
        [0.000000, 0.000000, 1.000000, 0.000000],
        [0.000000, 1.000000, 0.000000, 0.000000],
        [1.000000, 0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.000000, 1.000000]
    ])
    L = np.array([
        [1.000000, 0.000000, 0.000000, 0.000000],
        [-0.250000, 1.000000, 0.000000, 0.000000],
        [-0.250000, -0.818182, 1.000000, 0.000000],
        [0.750000, -0.818182, 0.853333, 1.000000]
    ])
    U = np.array([
        [-8.000000, -10.000000, -4.000000, 1.000000],
        [0.000000, -5.500000, -1.000000, -9.750000],
        [0.000000, 0.000000, -6.818182, -5.727273],
        [0.000000, 0.000000, 0.000000, 0.160000]
    ])
    # P = np.array([[0., 0., 1., 0.],
    #               [0., 1., 0., 0.],
    #               [1., 0., 0., 0.],
    #               [0., 0., 0., 1.]])
    # L = np.array([[1., 0., 0., 0.],
    #               [-0.25, 1., 0., 0.],
    #               [-0.25, -0.81818182, 1., 0.],
    #               [0.75, -0.81818182, 0.85333333, 1.]])
    # U = np.array([[-8., -10., -4., 1.],
    #               [0., -5.5, -1., -9.75],
    #               [0., 0., -6.81818182, -5.72727273],
    #               [0., 0., 0., 0.16]])
    b = [-91, -84, -64, -13]
    b = np.matmul(P, b)  # multiply matrix P with vector b
    print(b)
    y = forward_substitution(L, b)
    print(y)
    x = backward_substitution(U, y)
    return x


if __name__ == '__main__':
    print(plu_solve())
