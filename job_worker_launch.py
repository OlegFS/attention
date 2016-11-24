#!/bin/bash
for (( i=0; i<$1; i++ )); do
    /users-archive/neil/attention/job_worker_batch.py &
    sleep 2
done
