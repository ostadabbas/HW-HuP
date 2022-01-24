# ex basics
import numpy as np

## numpy
# try tht not euqal size list to np array
# li1 = [
# 	[2,3,4],
# 	[6,7]
# ]
# arr1 = np.array(li1)
# print(arr1)         #  not euqal size,get obj array,  each obj a list

# array mult symbol, point wise operator
arr1 = np.arange(6).reshape([2,3])
arr2 = np.arange(3).reshape([3,1])
arr3 = np.ones([2,3])*0.5
# print(arr1*arr2)        # can not broadcase
print(arr1*arr3)        # can not broadcase
