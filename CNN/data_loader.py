import numpy as np
import os
import pickle 

task = 'regression' # either 'classification' or 'regression'

def scale_data(data): # ppg scale
    rng = np.nanmax(data) - np.nanmin(data)
    minimum = np.nanmin(data)
    data = (data - minimum) / rng
    return data

def fill_2d_missing_with_previous(arr): # 결측값 이전 값으로 대체 
    # 결측값이 아닌 값들의 위치를 찾음
    valid = np.isnan(arr) == False
    
    # 누적 최대값을 사용하여 각 위치에서 마지막으로 관찰된 유효한 값을 채움
    filled = np.maximum.accumulate(valid, axis=1)

    # np.where를 사용하여 유효한 값과 결측값의 위치에 따라 값을 선택
    return np.where(filled, arr, np.nan)

if task == "classification":
    loaded_dataset = np.load(os.getcwd()+'/capstone/data/all_clf_arr.npz', allow_pickle = True)
else:
    loaded_dataset = np.load(os.getcwd()+'/capstone/data/all_reg_arr.npz', allow_pickle = True)
print("complete load")

x_arr = loaded_dataset['x_arr']
y_arr = loaded_dataset['y_arr']
c_arr = loaded_dataset['c_arr']
a_arr = loaded_dataset['a_arr']

x_arr = x_arr[~np.isnan(y_arr)]
c_arr = c_arr[~np.isnan(y_arr)]
a_arr = a_arr[~np.isnan(y_arr)]
y_arr = y_arr[~np.isnan(y_arr)]

x_arr = x_arr[:,:,1].reshape(-1, 1, 3000) # pytorch 를 위한 shape 으로 변경 
x_arr = fill_2d_missing_with_previous(x_arr)

print("complete fill_2d_missing_with_previous")

# x 에 nan 이 없는 것만 남기기 
idx = []
for i in range(len(x_arr)):
    if (np.isnan(x_arr[i]).sum() == 0)&(np.min(x_arr[i]) > 0):
        idx.append(i)
        
x_arr = x_arr[idx].astype(float)
y_arr = y_arr[idx].astype(float)
c_arr = c_arr[idx]

'''
# train, valid, test split (caseid 기준)
caseids = list(np.unique(c_arr))
nvalid = max(1, int(len(caseids) * 0.4))
ntest = max(1, int(len(caseids) * 0.2))

caseids_train = caseids[nvalid:]
caseids_valid = caseids[ntest:nvalid]
caseids_test = caseids[:ntest]
'''
print("complete remove nan")

if task == "classification":
    caseids_train, caseids_valid, caseids_test = pickle.load(open(os.getcwd()+'/capstone/data/clf_train_valid_test_caseids.pkl', 'rb'))
else:
    caseids_train, caseids_valid, caseids_test = pickle.load(open(os.getcwd()+'/capstone/data/regtrain_valid_test_caseids.pkl', 'rb'))

print("complete load reg_train_valid_test_caseids")

train_mask = np.isin(c_arr, caseids_train).flatten()
valid_mask = np.isin(c_arr, caseids_valid).flatten()
test_mask = np.isin(c_arr, caseids_test).flatten()

x_train = x_arr[train_mask]
y_train = y_arr[train_mask]
c_train = c_arr[train_mask]

x_valid = x_arr[valid_mask]
y_valid = y_arr[valid_mask]
c_valid = c_arr[valid_mask]

x_test = x_arr[test_mask]
y_test = y_arr[test_mask]
c_test = c_arr[test_mask]

print("complete data saperate")

for i in range(len(x_train)):
    x_train[i,0,:] = scale_data(x_train[i,0,:]) 

for i in range(len(x_valid)):
    x_valid[i,0,:] = scale_data(x_valid[i,0,:]) 

for i in range(len(x_test)):
    x_test[i,0,:] = scale_data(x_test[i,0,:])

print("complete data loading")

print("caseids train cnt: {}, caseids val cnt {}, caseids test cnt: {}".format(len(caseids_train), len(caseids_valid), len(caseids_test)))
print("samples train cnt: {}, samples val cnt: {}, samples test cnt: {}".format(len(x_train), len(x_valid), len(x_test)))

if task == "classification":
    prefix = "clf_"
else:
    prefix = "reg_"

pickle.dump((caseids_train), open(f"{os.getcwd()}/capstone/data/{prefix}caseids_train_vf.pkl", "wb"))
pickle.dump((caseids_valid), open(f"{os.getcwd()}/capstone/data/{prefix}caseids_valid_vf.pkl", "wb"))
pickle.dump((caseids_test), open(f"{os.getcwd()}/capstone/data/{prefix}caseids_test_vf.pkl", "wb"))
pickle.dump((x_train), open(f"{os.getcwd()}/capstone/data/{prefix}x_train_vf.pkl", "wb"))
pickle.dump((x_valid), open(f"{os.getcwd()}/capstone/data/{prefix}x_valid_vf.pkl", "wb"))
pickle.dump((x_test), open(f"{os.getcwd()}/capstone/data/{prefix}x_test_vf.pkl", "wb"))
pickle.dump((y_train), open(f"{os.getcwd()}/capstone/data/{prefix}y_train_vf.pkl", "wb"))
pickle.dump((y_valid), open(f"{os.getcwd()}/capstone/data/{prefix}y_valin_vf.pkl", "wb"))
pickle.dump((y_test), open(f"{os.getcwd()}/capstone/data/{prefix}y_test_vf.pkl", "wb"))
pickle.dump((c_train), open(f"{os.getcwd()}/capstone/data/{prefix}c_train_vf.pkl", "wb"))
pickle.dump((c_valid), open(f"{os.getcwd()}/capstone/data/{prefix}c_valid_vf.pkl", "wb"))
pickle.dump((c_test), open(f"{os.getcwd()}/capstone/data/{prefix}c_test_vf.pkl", "wb"))