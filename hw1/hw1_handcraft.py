import os
import pandas as pd
import numpy as np
import re
import math
import sys
import pickle

missing_values = ["n/a", "na", "--", "None", "NaN", "Na", "NA", "NR"]

#################
# training code #
#################

# x_y1 = pd.read_csv("./year1-data.csv", na_values = missing_values)
# x_y2 = pd.read_csv("./year2-data.csv", na_values = missing_values)

# xtrain = x_y1.append(x_y2)

# PM25 = xtrain[xtrain['測項'] == 'PM2.5']
# PM25 = PM25.drop(columns=['日期', '測項'])
# PM25_arr = PM25.values.flatten()

# x_train = []
# for i in range(len(PM25_arr)-9):
#     _temp = []
#     for j in range(10):
#         _temp.append(PM25_arr[i+j])
#     x_train.append(_temp)

# x_train_clear = []
# for i in x_train:
#     _flag = True
#     _temp = [1]
#     for j in i:
#         try:
#             if math.isnan(j):
#                 _flag = False
#                 break
#             else:
#                 _temp.append(j)
#         except:
#             try:
#                 _temp.append(int(re.compile("(\d+)").match(j).group(1)))
#             except:
#                 _flag = False
#                 break
#     if _flag:
#         x_train_clear.append(_temp)

# train_np_mat = np.asarray(x_train_clear)
# train_df_mat = pd.DataFrame(train_np_mat, columns=['b','x1','x2','x3','x4','x5','x6','x7','x8','x9','y'])
# x_train_np = train_df_mat.drop(columns=['y']).values

# iteration = 100000
# weight = np.asarray([1,1,1,1,1,1,1,1,1,1]).reshape(-1, 1)
# learning_rate = 0.000000001
# trans_mat = [1 for i in range(11)]

# def get_mat_1(weight):
#     ret = np.append(weight, -1)
#     ret = ret*2
#     return ret.reshape(-1,1)

# for i in range(iteration):
#     mat_1 = get_mat_1(weight)
#     grad =  np.dot(np.dot(train_np_mat, mat_1).T, x_train_np)
#     weight = weight - learning_rate * grad.reshape(-1,1)

# file = open('hw1_handcraft.pickle', 'wb')
# pickle.dump(weight, file)
# file.close()

test_data = pd.read_csv(sys.argv[1], na_values = missing_values)

xtest = test_data[test_data['測項'] == 'PM2.5']
x_test = xtest.drop(columns=['id','測項'])
x_test_clear = []
for i in x_test.values:
    _temp = [1]
    for j in i:
        try:
            _temp.append(int(re.compile("(\d+)").match(j).group(1)))
        except:
            _temp.append(j)
    x_test_clear.append(_temp)
x_test_np = np.asarray(x_test_clear)
x_test_df = pd.DataFrame(x_test_np)

with open('hw1_handcraft.pickle', 'rb') as file:
    weight = pickle.load(file)

y_test = np.dot(x_test_df, weight)

_id = ['id_'+str(i) for i in range(500)]
data = {'id':_id, 'value': y_test.flatten().tolist()}
output = pd.DataFrame(data)

output.to_csv(path_or_buf=sys.argv[2], index=False)

