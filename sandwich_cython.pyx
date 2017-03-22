#cython: boundscheck = True
#cython: cdivision = True

import numpy
cimport numpy


"""
=========
1D cases 
=========
"""


def filling_1d( numpy.ndarray[ numpy.float64_t, ndim = 1 ] cf ):
    """ Create : W D WT, where W is 1D DFT matrix. """
    cdef:
        numpy.ndarray[ numpy.float64_t, ndim = 2 ] C
        int s, i, j
    # setup
    s = cf.shape[ 0 ]
    C = numpy.empty( (s, s), dtype=cf.dtype )
    # fill in
    for i in range(s):
        for j in range(s):
            C[ i, j ] = cf[ (i - j) % s ]
    return C


def sandwich_1d( 
        numpy.ndarray[ numpy.float64_t, ndim = 1 ] cf,
        numpy.ndarray[ numpy.float64_t, ndim = 1 ] t ):
    """ Create :  T W D WT T """
    cdef:
        numpy.ndarray[ numpy.float64_t, ndim = 2 ] C
        int s, i, j
    # setup
    s = cf.shape[ 0 ]
    C = numpy.empty( (s, s), dtype=cf.dtype )
    # fill in
    for i in range(s):
        for j in range(s):
            C[ i, j ] = cf[ (i - j) % s ] * t[i] * t[j]
    return C


"""
========
2D cases
========
"""

def filling_2d( numpy.ndarray[ numpy.float64_t, ndim = 2 ] cf ):
    """ Create : W D WT, where W is 2D DFT matrix. """
    cdef:
        numpy.ndarray[ numpy.float64_t, ndim = 2 ] C
        int s0, s1, i, j, k, l
    # set up
    s0 = cf.shape[0]
    s1 = cf.shape[1]
    C = numpy.empty( ( s0 * s1, s0 * s1 ), dtype = cf.dtype )
    # fill in
    for i in range( s0 ):
        for j in range( s1 ):
            for k in range( s0 ):
                for l in range( s1 ):
                    C[ i * s1 + j, k * s1 + l ] = cf[ ( i - k ) % s0, ( j - l ) % s1 ] 
    return C


def sandwich_2d( 
        numpy.ndarray[ numpy.float64_t, ndim = 2 ] cf,
        numpy.ndarray[ numpy.float64_t, ndim = 2 ] t ):
    """ Create : W D WT, where W is 2D DFT matrix. """
    cdef:
        numpy.ndarray[ numpy.float64_t, ndim = 2 ] C
        int s0, s1, i, j, k, l
    # set up
    s0 = cf.shape[0]
    s1 = cf.shape[1]
    C = numpy.empty( ( s0 * s1, s0 * s1 ), dtype = cf.dtype )
    # fill in
    for i in range( s0 ):
        for j in range( s1 ):
            for k in range( s0 ):
                for l in range( s1 ):
                    C[ i * s1 + j, k * s1 + l ] = cf[ ( i - k ) % s0, ( j - l ) % s1 ] * t[i, j] * t[k, l]
    return C


def sandwich_2d_different_bread( 
        numpy.ndarray[ numpy.float64_t, ndim = 2 ] cf,
        numpy.ndarray[ numpy.float64_t, ndim = 2 ] t_left, 
        numpy.ndarray[ numpy.float64_t, ndim = 2 ] t_right ):
    """ Create : W D WT, where W is 2D DFT matrix. """
    cdef:
        numpy.ndarray[ numpy.float64_t, ndim = 2 ] C
        int s0, s1, i, j, k, l
    # set up
    s0 = cf.shape[0]
    s1 = cf.shape[1]
    C = numpy.empty( ( s0 * s1, s0 * s1 ), dtype = cf.dtype )
    # fill in
    for i in range( s0 ):
        for j in range( s1 ):
            for k in range( s0 ):
                for l in range( s1 ):
                    C[ i * s1 + j, k * s1 + l ] = cf[ ( i - k ) % s0, ( j - l ) % s1 ] * t_left[i, j] * t_right[k, l]
    return C

"""
========
Calzones
========
"""

def calzone( 
        numpy.ndarray[ numpy.float64_t, ndim = 1 ] re,
        numpy.ndarray[ numpy.float64_t, ndim = 1 ] im ):
    """ Create full calzone matrix. """
    cdef:
        numpy.ndarray[ numpy.float64_t, ndim = 2 ] C
        int N, halfN, i, j
    # initialise
    assert re.shape[0] == im.shape[0]
    N = re.shape[ 0 ]
    halfN = N / 2
    C = numpy.empty( (N, N), dtype = re.dtype )
    # DC column
    for i in range( halfN + 1 ):
        C[ i, 0 ] = re[ i ]
    for i in range( halfN + 1, N ):
        C[ i, 0 ] = im[ i - halfN ]
    # Nyquist column
    for i in range( halfN + 1 ):
        C[ i, halfN ] = re[ (i - halfN) % N ]
    for i in range( halfN + 1, N ):
        C[ i, halfN ] = im[ i ]
    # Top Left
    for i in range( halfN + 1 ):
        for j in range( 1, halfN ):
            C[ i, j ] = re[ (i - j) % N ] + re[ (i + j) % N ]
    # Top Right
    for i in range( halfN + 1 ):
        for j in range( halfN + 1, N ):
            C[ i, j ] = -im[ (i - j + halfN) % N ] + im[ (i - halfN + j) % N ]
    # Bottom Left
    for i in range( halfN + 1, N ):
        for j in range( 1, halfN ):
            C[ i, j ] = im[ (i - halfN - j) % N ] + im[ (i - halfN + j) % N ]
    # Bottom Right
    for i in range( halfN + 1, N ):
        for j in range( halfN + 1, N ):
            C[ i, j ] = re[ (i - j) % N ] - re[ (i + j) % N ]
    return C


def calzino(
        numpy.ndarray[ numpy.float64_t, ndim = 1 ] re,
        numpy.ndarray[ numpy.float64_t, ndim = 1 ] im,
        int F ):
    """ Create only the appropriate rows/columns of the calzone matrix. """
    cdef:
        numpy.ndarray[ numpy.float64_t, ndim = 2 ] C
        int N, halfN, row, col
    # initialise
    assert re.shape[0] == im.shape[0]
    N = re.shape[ 0 ]
    assert 2 * F + 1 < N
    halfN = N / 2
    C = numpy.empty( (2*F + 1, 2*F + 1), dtype = re.dtype )
    # DC column
    for row in range( F + 1 ): # i in range( halfN + 1 ):
        # i = row
        C[ row, 0 ] = re[ row ] # i
    for row in range( F + 1, 2*F + 1 ): # i in range( halfN + 1, N ):
        # i = row - F + halfN
        C[ row, 0 ] = im[ row - F ] # i - halfN
    # No Nyquist column, as this is low pass
    # Top Left
    for row in range( F + 1 ):
        for col in range( 1, F + 1 ):
            # i = row,  j = col
            # C[ i, j ] = re[ (i - j) % N ] + re[ (i + j) % N ]
            C[ row, col ] = re[ (row - col) % N ] + re[ (row + col) % N ]
    # Top Right
    for row in range( F + 1 ):
        for col in range( F + 1, 2 * F + 1 ):
            # i = row,  j = col - F + halfN
            # C[ i, j ] = -im[ (i - j + halfN) % N ] + im[ (i - halfN + j) % N ]
            C[ row, col ] = (
                    -im[ (row - col + F) % N ] + im[ (row + col - F) % N ] )
    # Bottom Left
    for row in range( F + 1, 2 * F + 1 ):
        for col in range( 1, F + 1 ):
            # i = row - F + halfN,  j = col
            # C[ i, j ] = im[ (i - halfN - j) % N ] + im[ (i - halfN + j) % N ]
            C[ row, col ] = ( 
                    im[ (row - F - col) % N ] + im[ (row - F + col) % N ] )
    # Bottom Right
    for row in range( F + 1, 2 * F + 1 ):
        for col in range( F + 1, 2 * F + 1 ):
            # i = row - F + halfN,  j = col - F + halfN
            # C[ i, j ] = re[ (i - j) % N ] - re[ (i + j) % N ]
            C[ row, col ] = re[ (row - col) % N ] - re[ (row - 2*F + col) % N ]
    return C
    
def calzinotto(
        numpy.ndarray[ numpy.float64_t, ndim = 1 ] re,
        numpy.ndarray[ numpy.float64_t, ndim = 1 ] im,
        int F_rows, int F_cols ):
    """ Create only the appropriate rows/columns of the calzone matrix. """
    cdef:
        numpy.ndarray[ numpy.float64_t, ndim = 2 ] C
        int N, halfN, row, col
    # initialise
    assert re.shape[0] == im.shape[0]
    N = re.shape[ 0 ]
    assert 2 * F_rows + 1 < N
    assert 2 * F_cols + 1 < N
    halfN = N / 2
    C = numpy.empty( (2*F_rows + 1, 2*F_cols + 1), dtype = re.dtype )
    # DC column
    for row in range( F_rows + 1 ): # i in range( halfN + 1 ):
        # i = row
        C[ row, 0 ] = re[ row ] # i
    for row in range( F_rows + 1, 2*F_rows + 1 ): # i in range( halfN + 1, N ):
        # i = row - F + halfN
        C[ row, 0 ] = im[ row - F_rows ] # i - halfN
    # No Nyquist column, as this is low pass
    # Top Left
    for row in range( F_rows + 1 ):
        for col in range( 1, F_cols + 1 ):
            # i = row,  j = col
            # C[ i, j ] = re[ (i - j) % N ] + re[ (i + j) % N ]
            C[ row, col ] = re[ (row - col) % N ] + re[ (row + col) % N ]
    # Top Right
    for row in range( F_rows + 1 ):
        for col in range( F_cols + 1, 2 * F_cols + 1 ):
            # i = row,  j = col - F_cols + halfN
            # C[ i, j ] = -im[ (i - j + halfN) % N ] + im[ (i - halfN + j) % N ]
            C[ row, col ] = (
                    -im[ (row - col + F_cols) % N ] 
                    + im[ (row + col - F_cols) % N ] )
    # Bottom Left
    for row in range( F_rows + 1, 2 * F_rows + 1 ):
        for col in range( 1, F_cols + 1 ):
            # i = row - F_rows + halfN,  j = col
            # C[ i, j ] = im[ (i - halfN - j) % N ] + im[ (i - halfN + j) % N ]
            C[ row, col ] = ( 
                    im[ (row - F_rows - col) % N ] 
                    + im[ (row - F_rows + col) % N ] )
    # Bottom Right
    for row in range( F_rows + 1, 2 * F_rows + 1 ):
        for col in range( F_cols + 1, 2 * F_cols + 1 ):
            # i = row - F_rows + halfN,  j = col - F_cols + halfN
            # C[ i, j ] = re[ (i - j) % N ] - re[ (i + j) % N ]
            C[ row, col ] = (
                    re[ (row - F_rows - col + F_cols) % N ] 
                    -re[ (row - F_rows + col - F_cols) % N ] )
    return C
    
