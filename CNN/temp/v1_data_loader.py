import numpy as np
import pandas as pd
import os
import pickle
import vitaldb
import random
from pyvital import arr
import time
import multiprocessing

ECG = 'SNUADC/ECG_II'
PPG = 'SNUADC/PLETH'
ART = 'SNUADC/ART'
MBP = 'Solar8000/ART_MBP'
CO2 = 'Primus/CO2'  # 62.5hz 로 다른 waveform 과 주파수가 다름 

def check_hypotension(map_values, threshold=65):
    consecutive_count = 0
    for map_value in map_values:
        if map_value <= threshold:
            consecutive_count += 1
            if consecutive_count >= len(map_values):
                return True
        else:
            consecutive_count = 0
    return False

def check_non_hypotension(map_values, threshold=65):
    consecutive_count = 0
    for map_value in map_values:
        if map_value > threshold:
            consecutive_count += 1
            if consecutive_count >= len(map_values): 
                return True
        else:
            consecutive_count = 0
    return False

def data_loader(caseid, df_case):
    opstart = df_case['opstart'].values[0]
    opend = df_case['opend'].values[0]
    
    demo = df_case[['age', 'sex', 'weight', 'height', 'asa']].values
    
    MINUTES_AHEAD = 5*60
    SRATE = 100
    INSEC = 30  # input 길이 (논문과 같음)
    SW_SIZE = 1 * 60 # sliding window size (논문에 따로 정의되어 있지 않음)

    vals = vitaldb.load_case(caseid, [ECG, PPG, CO2, MBP], 1/SRATE)
    vals[:,3] = arr.replace_undefined(vals[:,3])
    vals = vals[opstart * SRATE:opend * SRATE]
    
    x = []
    y = []
    c = []
    a = []
    for i in range(0, len(vals) - (SRATE * (INSEC + MINUTES_AHEAD) + 1), SW_SIZE*SRATE):
        segx = vals[i:i + SRATE * INSEC, :3]  
        segy_1min = vals[i + SRATE * (INSEC + MINUTES_AHEAD) + 1:(i + SRATE * (INSEC + MINUTES_AHEAD) + 1)+(SRATE * 60), 3]
        segy_20min = vals[i + SRATE * (INSEC + MINUTES_AHEAD) + 1:(i + SRATE * (INSEC + MINUTES_AHEAD) + 1)+(SRATE * 20 * 60), 3]
        
        if check_hypotension(segy_1min):
            segy = 1
        elif check_non_hypotension(segy_20min):
            segy = 0
        else:
            segy = np.nan

        x.append(segx)
        y.append(segy)
        c.append(caseid)
        a.append(demo)
                
    if len(x) > 0:
        print(caseid)
        
        ret = (np.array(x), np.array(y), np.array(c), np.array(a)) 
        pickle.dump((ret), open(f"{os.getcwd()}/capstone/data/md_hypo/minutes5_clf/{caseid}_vf.pkl", "wb"))

def data_loader_reg(caseid, df_case):
    opstart = df_case['opstart'].values[0]
    opend = df_case['opend'].values[0]
    
    demo = df_case[['age', 'sex', 'weight', 'height', 'asa']].values
    
    MINUTES_AHEAD = 5*60
    SRATE = 100
    INSEC = 30  # input 길이 (논문과 같음)
    SW_SIZE = 5  # sliding window size (논문에 따로 정의되어 있지 않음)

    vals = vitaldb.load_case(caseid, [ECG, PPG, ART, CO2, MBP], 1/SRATE)
    vals[:,2] = arr.replace_undefined(vals[:,2])
    vals = vals[opstart * SRATE:opend * SRATE]
    
    # 20sec (20 00) - 5min (300 00) - 1min (60 00) = 38000 sample
    x = []
    y = []
    c = []
    a = []
    for i in range(0, len(vals) - (SRATE * (INSEC + MINUTES_AHEAD) + 1), SRATE * SW_SIZE):
        segx = vals[i:i + SRATE * INSEC, :4]  
        segy = vals[i + SRATE * (INSEC + MINUTES_AHEAD) + 1, 4]
        
        if segy < 20 or segy > 200:
            continue

        '''
        # maic 대회 참고하여 전처리 
        if np.mean(np.isnan(segx)) > 0 or \
            np.mean(np.isnan(segy)) > 0 or \
            np.max(segy) > 200 or np.min(segy) < 20 or \
            np.max(segy) - np.min(segy) < 30 or \
            (np.abs(np.diff(segy[~np.isnan(segy)])) > 30).any():
            i += SRATE  # 1 sec 씩 전진
            continue
        '''    
        x.append(segx)
        y.append(segy)
        c.append(caseid)
        a.append(demo)
                
    if len(x) > 0:
        print(caseid)
        ret = (np.array(x), np.array(y), np.array(c), np.array(a)) 
        pickle.dump((ret), open(f"{os.getcwd()}/capstone/data/md_hypo/minutes5_reg/{caseid}_vf.pkl", "wb"))


def main_classification():
    path = os.getcwd()
    case_dir = f"{path}/capstone/data/md_hypo/minutes5_clf/"  
    if not ( os.path.isdir( case_dir ) ):
        os.makedirs ( os.path.join ( case_dir ) )

    df_trks = pd.read_csv('https://api.vitaldb.net/trks')  # 트랙 목록
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")  # 임상 정보

    caseids = list(
    set(df_trks[df_trks['tname'] == MBP]['caseid']) &
    set(df_trks[df_trks['tname'] == CO2]['caseid']) &
    set(df_trks[df_trks['tname'] == ART]['caseid']) &
    set(df_trks[df_trks['tname'] == ECG]['caseid']) &
    set(df_trks[df_trks['tname'] == PPG]['caseid']))
    
    df_cases = df_cases[(df_cases['caseid'].isin(caseids))&(df_cases['age']>=18)&(df_cases['death_inhosp']!=1)]
    caseids = list(df_cases['caseid'].values)
    
    already_caseids = os.listdir(os.getcwd()+'/capstone/data/md_hypo/minutes5_clf/')
    already_caseids = [int(caseid.replace('_vf.pkl','')) for caseid in already_caseids]
    
    caseids = list(set(caseids) - set(already_caseids))
    print(len(caseids))
    
    start_time = time.time()
    n_process = 90

    manager = multiprocessing.Manager() 
    d = manager.dict() # shared dictionary

    pool = multiprocessing.Pool(processes=n_process)
    for caseid in caseids:
        pool.apply_async(data_loader, (caseid, df_cases[df_cases['caseid']==caseid]))

    pool.close()
    pool.join()
    
    print("=== %s seconds ===" % (time.time() - start_time))   # 5개 266초  

def main_regression():
    path = os.getcwd()
    case_dir = f"{path}/capstone/data/md_hypo/minutes5_reg/"  
    if not ( os.path.isdir( case_dir ) ):
        os.makedirs ( os.path.join ( case_dir ) )

    df_trks = pd.read_csv('https://api.vitaldb.net/trks')  # 트랙 목록
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")  # 임상 정보

    caseids = list(
    set(df_trks[df_trks['tname'] == MBP]['caseid']) &
    set(df_trks[df_trks['tname'] == CO2]['caseid']) &
    set(df_trks[df_trks['tname'] == ART]['caseid']) &
    set(df_trks[df_trks['tname'] == ECG]['caseid']) &
    set(df_trks[df_trks['tname'] == PPG]['caseid']))
    
    df_cases = df_cases[(df_cases['caseid'].isin(caseids))&(df_cases['age']>=18)&(df_cases['death_inhosp']!=1)]
    caseids = list(df_cases['caseid'].values)
    
    already_caseids = os.listdir(os.getcwd()+'/capstone/data/md_hypo/minutes5_reg/')
    already_caseids = [int(caseid.replace('_vf.pkl','')) for caseid in already_caseids]
    
    caseids = list(set(caseids) - set(already_caseids))
    print(len(caseids))
    
    start_time = time.time()
    n_process = 80

    manager = multiprocessing.Manager() 
    d = manager.dict() # shared dictionary

    pool = multiprocessing.Pool(processes=n_process)
    for caseid in caseids:
        pool.apply_async(data_loader_reg, (caseid, df_cases[df_cases['caseid']==caseid]))

    pool.close()
    pool.join()
    
    print("=== %s seconds ===" % (time.time() - start_time))   # 5개 266초  

if __name__ == '__main__':
    # main_classification()
    # main_regression()

    path = os.getcwd()
    loaded_dataset = np.load(f"{path}/capstone/data/md_hypo/minutes5_reg_/1_vf.pkl", allow_pickle = True)
    print(loaded_dataset)