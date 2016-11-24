""" Setup """

from rq import Queue
from job_worker_connect import conn
import progress

from numpy.random import permutation

q = Queue( connection=conn )


""" Queue jobs """

#import A3_fit_CGL as A
#import B1_mod_tau as A
#import B2_mod_fit as A
#import B3_mod_cv as A
#import B4_mod_cv_sharedC as A
#import A8_fit_sharedC as A
#import C1
#import C2
import D3_time_course as A

for script in [A]:
    for repeat in range(3):
        for j in progress.dots( permutation( script.jobs ), 
                '%s: adding jobs (%d)' % (script.__name__, repeat) ):
            func = j[0]
            args = tuple([a for a in j[1:]])
            result = q.enqueue_call( func, args=args, timeout=60*60*4 )
