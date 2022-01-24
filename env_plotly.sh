#!/bin/bash
# setup discovery env.  load modules gcc, anaconda3/2020, cuda9.0.  activate env
echo loading plotly container env ...
bash /shared/container_repository/plotly-ds/run_container_plotly.sh
source /opt/miniconda3/bin/activate