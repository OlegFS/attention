import numba
from numpy import empty

f64 = numba.float64
c128 = numba.complex128
i64 = numba.int64


"""
========
1D cases
========
"""


@numba.jit( f64[ :, : ]( f64[ : ] ) )
def filling_1d( cf ):
    """ Create : W D WT, where W is 1D DFT matrix. """
    s = cf.shape
    c = empty( ( s[ 0 ], s[ 0 ] ), dtype = cf.dtype )
    for i in range( s[ 0 ] ):
        for j in range( s[ 0 ] ):
            c[ i, j ] = cf[ ( i - j ) % s[ 0 ] ]
    return c


@numba.jit( c128[ :, : ]( c128[ : ] ) )
def filling_1d_complex( cf ):
    s = cf.shape
    c = empty( ( s[ 0 ], s[ 0 ] ), dtype = cf.dtype )
    for i in range( s[ 0 ] ):
        for j in range( s[ 0 ] ):
            c[ i, j ] = cf[ ( i - j ) % s[ 0 ] ]
    return c


@numba.jit( f64[ :, : ]( f64[ : ], f64[ : ] ) )
def sandwich_1d( cf, t ):
    """ Create :  T W D WT T """
    s = cf.shape
    C = empty( ( s[ 0 ], s[ 0 ] ), dtype = cf.dtype )
    for i in range( s[ 0 ] ):
        for j in range( s[ 0 ] ):
            C[ i, j ] = cf[ ( i - j ) % s[ 0 ] ] * t[ i ] * t[ j ]
    return C

"""
========
2D cases
========
"""

@numba.jit( f64[ :, : ]( f64[ :, : ] ) )
def filling_2d( cf ):
    """ Create : W D WT, where W is 2D DFT matrix. """
    s = cf.shape
    C = empty( ( s[ 0 ] * s[ 1 ], s[ 0 ] * s[ 1 ] ), dtype = cf.dtype )
    for i in range( s[ 0 ] ):
        for j in range( s[ 1 ] ):
            for k in range( s[ 0 ] ):
                for l in range( s[ 1 ] ):
                    C[ i * s[ 1 ] + j, k * s[ 1 ] + l ] = cf[ ( i - k ) % s[ 0 ], ( j - l ) % s[ 1 ] ] 
    return C


@numba.jit( f64[ :, : ]( f64[ :, : ], f64[ :, : ] ) )
def sandwich_2d( cf, t ):
    """ Create : T W D WT T, where W is 2D DFT matrix. """
    s = cf.shape
    C = empty( ( s[ 0 ] * s[ 1 ], s[ 0 ] * s[ 1 ] ), dtype = cf.dtype )
    for i in range( s[ 0 ] ):
        for j in range( s[ 1 ] ):
            for k in range( s[ 0 ] ):
                for l in range( s[ 1 ] ):
                    C[ i * s[ 1 ] + j, k * s[ 1 ] + l ] = cf[ ( i - k ) % s[ 0 ], ( j - l ) % s[ 1 ] ] * t[ i, j ] * t[ k, l ]
    return C



"""
========
Calzones
========
"""

@numba.jit( f64[:, :]( f64[:], f64[:] ) )
def calzone( re, im ):
    # sizes
    N = re.shape[ 0 ]
    halfN = N / 2
    # initialise
    C = empty( (N, N), dtype = re.dtype )
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
   

@numba.jit( f64[:, :]( f64[:], f64[:], i64 ) )
def calzino( re, im, max_idx ):
    # sizes
    N = re.shape[ 0 ]
    halfN = N / 2
    F = max_idx
    # initialise
    C = empty( (2*F + 1, 2*F + 1), dtype = re.dtype )
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
    
