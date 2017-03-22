import numpy
import pyximport

pyximport.install( 
        reload_support = True,
        setup_args = { 'include_dirs' : [ numpy.get_include() ] } )

from sandwich_cython import *
