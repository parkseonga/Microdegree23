import numpy as np
import os
import pickle
import multiprocessing

def load_data(clf = True):
    path = os.getcwd()
    if clf:
        loaded_dataset = np.load(f"{path}/capstone/data/all_clf_arr.npz", allow_pickle = True)
    else:
        loaded_dataset = np.load(f"{path}/capstone/data/all_reg_arr.npz", allow_pickle = True)
    print('== load dataset file complete ==')

    # x_arr = loaded_dataset['x_arr']
    # y_arr = loaded_dataset['y_arr']
    # c_arr = loaded_dataset['c_arr']
    # a_arr = loaded_dataset['a_arr']

    # print("x count : " + str(len(x_arr)) + " / y count : " + str(len(y_arr)) + " / c count : " + str(len(c_arr)) + " / a count : " + str(len(a_arr)))

    return loaded_dataset

def analysis_data(loaded_dataset):
    x_arr = loaded_dataset['x_arr']
    y_arr = loaded_dataset['y_arr']
    c_arr = loaded_dataset['c_arr']
    a_arr = loaded_dataset['a_arr']

    print(len(x_arr))
    #y_arr 각 값의 개수 찾기
    unique_values, counts = np.unique(y_arr, return_counts=True)
    print('== unique values find complete ==')
    print(unique_values)
    print(counts)
    unique_counts = dict(zip(unique_values, counts))
    print('== unique counts find complete ==')
    print(unique_counts)

def data_0(loaded_dataset, clf = True):
    print(loaded_dataset['c_arr'][0])
    print(loaded_dataset['x_arr'][0])
    print(loaded_dataset['y_arr'][0])
    print(loaded_dataset['a_arr'][0])

def filter_x(x):
    for i in x:
        for j in i:
            if np.isnan(j):
                return True
    return False

def filter_y(y):
    if np.isnan(y):
        return True
    return False

def make_data(i, ret, clf = True):
    path = os.getcwd()
    foldername = "minutes5_clf"
    if clf == False:
        foldername = "minutes5_reg"

    if filter_x(ret[1]):
        print("X is nan, Pass~~")
        print(i)
        print((ret[0], ret[1], ret[2], ret[3]))
        pickle.dump((ret), open(f"{path}/capstone/data/md_hypo/{foldername}/pass_x/{ret[0]}/{i}_vf.pkl", "wb"))
        return
    
    if filter_y(ret[2]):
        print("Y is nan, Pass~~")
        print(i)
        print((ret[0], ret[1], ret[2], ret[3]))
        pickle.dump((ret), open(f"{path}/capstone/data/md_hypo/{foldername}/pass_y/{ret[0]}/{i}_vf.pkl", "wb"))
        return
    
    print("secced")
    print(i)
    print((ret[0], ret[1], ret[2], ret[3]))
    pickle.dump((ret), open(f"{path}/capstone/data/md_hypo/{foldername}/{ret[0]}/{i}_vf.pkl", "wb"))

def remake(loaded_dataset, clf = True):
    path = os.getcwd()
    foldername = "minutes5_clf"
    if clf == False:
        foldername = "minutes5_reg"

    x_arr = loaded_dataset['x_arr']
    y_arr = loaded_dataset['y_arr']
    c_arr = loaded_dataset['c_arr']
    a_arr = loaded_dataset['a_arr']

    case_dir = f"{path}/capstone/data/md_hypo/{foldername}/"  
    if not ( os.path.isdir( case_dir ) ):
        os.makedirs ( os.path.join ( case_dir ) )

    case_dir = f"{path}/capstone/data/md_hypo/{foldername}/pass_x/"  
    if not ( os.path.isdir( case_dir ) ):
        os.makedirs ( os.path.join ( case_dir ) )

    case_dir = f"{path}/capstone/data/md_hypo/{foldername}/pass_y/"  
    if not ( os.path.isdir( case_dir ) ):
        os.makedirs ( os.path.join ( case_dir ) )
        
    for i in range(len(c_arr)):
        case_dir = f"{path}/capstone/data/md_hypo/{foldername}/{c_arr[i]}/"  
        if not ( os.path.isdir( case_dir ) ):
            os.makedirs ( os.path.join ( case_dir ) )

        case_dir = f"{path}/capstone/data/md_hypo/{foldername}/pass_x/{c_arr[i]}/"  
        if not ( os.path.isdir( case_dir ) ):
            os.makedirs ( os.path.join ( case_dir ) )
        
        case_dir = f"{path}/capstone/data/md_hypo/{foldername}/pass_y/{c_arr[i]}/"  
        if not ( os.path.isdir( case_dir ) ):
            os.makedirs ( os.path.join ( case_dir ) )

    n_process = 20
    manager = multiprocessing.Manager() 
    d = manager.dict() # shared dictionary

    pool = multiprocessing.Pool(processes=n_process)
    for i in range(len(c_arr)):
        pool.apply_async(make_data, (i, (c_arr[i], x_arr[i], y_arr[i], a_arr[i]), clf))

    pool.close()
    pool.join()

    # for i in range(len(c_arr)):
    #     print((c_arr[i], x_arr[i], y_arr[i], a_arr[i]))
    #     make_data(i, (c_arr[i], x_arr[i], y_arr[i], a_arr[i]), clf)
    #     print(f"{i}/{len(c_arr)}")

    foldername = "minutes5_clf"
    if clf == False:
        foldername = "minutes5_reg"
    print(f"{foldername} : make {len(c_arr)}")

def check_count(clf = True):
    path = os.getcwd()
    foldername = "minutes5_clf"
    if clf == False:
        foldername = "minutes5_reg"
    
    case_dir = f"{path}/capstone/data/md_hypo/{foldername}/"  
    case_list = os.listdir(case_dir)
    if '.DS_Store' in case_list:
        case_list.remove('.DS_Store')
    total_item = []
    for c in case_list:
        item_list = os.listdir(case_dir + c)
        if '.DS_Store' in case_dir:
            case_dir.remove('.DS_Store')
        total_item += item_list

    px_case_dir = f"{path}/capstone/data/md_hypo/{foldername}/pass_x/"  
    px_case_list = os.listdir(px_case_dir)
    if '.DS_Store' in px_case_list:
        px_case_list.remove('.DS_Store')
    px_total_item = []
    for c in px_case_list:
        item_list = os.listdir(px_case_dir + c)
        if '.DS_Store' in px_case_dir:
            px_case_dir.remove('.DS_Store')
        px_total_item += item_list

    py_case_dir = f"{path}/capstone/data/md_hypo/{foldername}/pass_y/"  
    py_case_list = os.listdir(py_case_dir)
    if '.DS_Store' in py_case_list:
        py_case_list.remove('.DS_Store')
    py_total_item = []
    for c in py_case_list:
        item_list = os.listdir(py_case_dir + c)
        if '.DS_Store' in item_list:
            item_list.remove('.DS_Store')
        py_total_item += item_list

    print(f"make_item count : {len(total_item)}, pass_x : {len(px_total_item)}, pass_y : {len(py_total_item)}")

if __name__ == '__main__':
    # loaded_dataset = load_data() 
    # remake(loaded_dataset)
    # loaded_dataset = load_data(False) 
    # remake(loaded_dataset, False)

    # check_count()
    # check_count(False)

    # loaded_dataset = load_data() 
    # data_0(loaded_dataset)
    # loaded_dataset = load_data(False) 
    # data_0(loaded_dataset, False)

    path = os.getcwd()
    loaded_dataset = np.load(f"{path}/capstone/data/md_hypo/temp_minutes5_reg/3_vf.pkl", allow_pickle = True)
    print(loaded_dataset)


# ex...
# 3681
# [[-2.88494006e-02  3.12346001e+01  1.39999998e+00]
#  [-2.88494006e-02  3.12346001e+01  1.29999995e+00]
#  [-2.88494006e-02  3.12346001e+01  1.29999995e+00]
#  ...
#  [-2.88494006e-02  4.26890984e+01  1.50000000e+00]
#  [-2.88494006e-02  4.54539986e+01  1.50000000e+00]
#  [-2.88494006e-02  4.90088005e+01  1.50000000e+00]]
# 0.0
# [[62.0 'M' 78.9 175.4 2.0]]