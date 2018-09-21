#!/bin/bash

for DIM in 3 4 5 6 7 8 9 
do
  for VAL in {1..50}
  do
    qsub -v var1=0.3,var2=4,var3=$VAL,var4=$DIM jobscript
  done
done
