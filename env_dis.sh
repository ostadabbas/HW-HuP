#!/bin/bash
# setup discovery env.  load modules gcc, anaconda3/2020, cuda9.0.  activate env
echo loading disocvery env ...
module load discovery/2019-02-21
module load anaconda3/2020.02
module load gcc/5.5.0
module load cuda/9.0

source activate smplify2