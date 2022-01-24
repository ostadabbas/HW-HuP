#!/bin/bash
# do a basic call test for a simple
#
#python ex_call_pkg.py
#echo ${1:-haha}
#echo $1   # is empty
# there is input, for sub 1, if we give | directly  what happened, rst: bar is deemed as input nothing happened
# will not print if | as other's stdin, nothing printed
sh scripts/ex_sub1.sh "h" | echo
# will print 2 h
sh scripts/ex_sub1.sh "h" | xars echo