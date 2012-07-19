#!/bin/bash

mpiexec -np 128 -hostfile  /etc/hostfile --bynode -x OMP_NUM_THREADS=1   -x LD_LIBRARY_PATH ./qr_test.x -p 8 -q 16 -r 2000 20000 2000 -nb 100

