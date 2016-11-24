""" Setup """

from rq import Queue
from job_worker_connect import conn
import progress

from numpy.random import permutation

q = Queue( connection=conn )


""" Queue jobs """

import C1_7
import C1_8
import C1_9

for j in progress.dots(C1_7.jobs, 'adding jobs'):
    func = j[0]
    args = tuple([a for a in j[1:]])
    result = q.enqueue_call( func, args=args, timeout=60*60*4 )
for j in progress.dots(C1_8.jobs, 'adding jobs'):
    func = j[0]
    args = tuple([a for a in j[1:]])
    result = q.enqueue_call( func, args=args, timeout=60*60*4 )
for j in progress.dots(C1_9.jobs, 'adding jobs'):
    func = j[0]
    args = tuple([a for a in j[1:]])
    result = q.enqueue_call( func, args=args, timeout=60*60*4 )
