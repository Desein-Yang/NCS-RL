#!/bin/bash

# Example
# mpirun -np 31 python main.py -e 26 -g MontezumaRevenge -r run1

# run id start 
i=$1
# run id end, 不包括end
end=$2

while [ $i -lt $end ]
do
    mpirun -np 40 python main.py -e 20 -g MontezumaRevenge -r run${i}
    i=$((i+1))
done
