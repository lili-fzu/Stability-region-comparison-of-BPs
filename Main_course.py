# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:47:57 2023

@author: Lin and Lily
"""

import numpy as np
import os
import pandas as pd
import copy
import pickle
import time
from arch.unitroot import DFGLS
from statsmodels.tsa.stattools import adfuller
######Lin
import math
import datetime
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

def Arrival_gen(Arrival_origin, T):
    Arrival = copy.deepcopy(empty2)
    for index in ['N','E']:
        for Dir in ['T']:
            Arrival[index][Dir] = Arrival_origin[index][Dir][Arrival_origin[index][Dir]['Arrival_time'] <= T]
    return Arrival

def Arrival_generate(name,Headway,T,D,isload,arrival_last): # 生成 到达车辆
    '''
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir+'\\arrival\\', name+'.p')
    if os.path.isfile(file_path): # if the file exists, load it
        isload = 1
    else: #generate new file otherwise
        isload = 0
    '''
    if isload == 1: #加载数据
        Arrival = pickle.load(open('arrival/'+name+'.p', "rb"))
    elif isload == 0: #生成数据
        Arrival = copy.deepcopy(empty2)
        for index in ['N','E']:
            for Dir in ['T']:
                v_type = [1]*len(Headway[index][Dir]['veh_type'].tolist())
                d = D[index][Dir]
                if d!=0:
                    t = 0
                    arrival_interval = 3600/d
                    while t < T:
                        t_temp = np.random.exponential(scale=arrival_interval) # 到达间隔
                        t += t_temp; # 到达时间
                        veh_type = v_type[int(np.random.rand()*len(v_type))] # 车辆类型
                        Arrival[index][Dir].loc[len(Arrival[index][Dir])] = [t, t_temp, veh_type]
    elif isload == 0.5:
        Arrival = arrival_last
        for index in ['N','E']:
            for Dir in ['T']:
                v_type = [1]*len(Headway[index][Dir]['veh_type'].tolist())
                d = D[index][Dir]
                if d!=0:
                    arrival_interval = 3600/d
                    if len(Arrival[index][Dir])>0:
                        t = max(Arrival[index][Dir]['Arrival_time'])
                        Arrival[index][Dir] = Arrival[index][Dir].reset_index(drop=True)
                    else:
                        t = T_step + delta_t
                    while t < T:
                        t_temp = np.random.exponential(scale=arrival_interval) # 到达间隔
                        t += t_temp; # 到达时间
                        veh_type = v_type[int(np.random.rand()*len(v_type))] # 车辆类型
                        Arrival[index][Dir].loc[len(Arrival[index][Dir])] = [t, t_temp, veh_type]
    return Arrival

def dir2phase(index,Dir,Phase): # 流向转phase
    phase_all = []
    for phase_seq in range(len(Phase)):
        for j in Phase[phase_seq]:
            index0,Dir0 = j2Dir(j)
            if index==index0 and Dir==Dir0:
                phase_all.append(phase_seq) 
    return phase_all

def ds_ratio_estimate(SFR_mean, delta_t, lost_time, D):
    supply_T = 0
    demand_T = 0
    for index in ['N','E']:
        supply_T += SFR_mean[index]['T']
        demand_T += D[index]['T']
    mean_supply_T = supply_T / 4 * delta_t / (delta_t + lost_time)
    ds_ratio_T = demand_T / mean_supply_T
    return ds_ratio_T 
    
def flow_discount_change(D0, flow_discount):
    D = copy.deepcopy(empty1)
    for index, data in D.items():
        for Dir in data.keys():
            if Dir == 'L':
                D[index][Dir] = D0[index][Dir] * flow_discount
            else:
                D[index][Dir] = (D0[index]['T1'] + D0[index]['T2']) * flow_discount
    return D

def flow_add_change(D0, delta_demand):
    D = copy.deepcopy(empty1)
    for index, data in D.items():
        for Dir in data.keys():
            if Dir == 'L':
                D[index][Dir] = max(D0[index][Dir] + delta_demand,0)
            else:
                D[index][Dir] = max((D0[index]['T1'] + D0[index]['T2']) + delta_demand,0)
    return D

def get_delay(T_step, Queue, Departure, queue_length, Delay, queue_list, delay_list):
    # 统计实时排队长度等数据
    mean_delay = None
    mean_queue_length = []
    if T_step > warm_up_time:
        for index in ['N','E']:
            for Dir in ['T']:
                queue_length += len(Queue[index][Dir])
    
    if int((T_step - warm_up_time) / statistic_interval) == len(queue_list) + 1:
        cycle_no = statistic_interval / (delta_t ) # + lost_time
        begin_T = T_step - statistic_interval
        end_T = T_step
        total_delay = 0
        total_veh = 0
        for index in ['N','E']:
            for Dir in ['T']:
                total_veh_dir = 0
                total_delay_dir = 0
                total_veh_dir += len(Queue[index][Dir])
                if len(Queue[index][Dir]) > 0:
                    for k in Queue[index][Dir].index:
                        total_delay_dir += T_step + delta_t  - Queue[index][Dir].loc[k,'Arrival_time'] # + lost_time
                temp = Departure[index][Dir]
                depart_temp = temp[(temp['Arrival_time'] >= begin_T) & (temp['Arrival_time'] < end_T)]
                total_veh_dir += len(depart_temp)
                if len(depart_temp) > 0:
                    for k in depart_temp.index:
                        total_delay_dir += depart_temp.loc[k,'Delay']
                mean_delay_dir = total_delay_dir / total_veh_dir if total_veh_dir > 0 else 0
                Delay[index][Dir].loc[len(Delay[index][Dir])] = [T_step, mean_delay_dir]
                total_delay += total_delay_dir
                total_veh += total_veh_dir
        mean_delay = total_delay / total_veh if total_veh > 0 else 0
        mean_queue_length = queue_length / cycle_no
        queue_list.append(int(mean_queue_length))
        delay_list.append(mean_delay)
        queue_length = 0
        if int(T_step%360) == 0:
            print('t=',int(T_step), 'delay=','{:.1f}'.format(mean_delay), 'Queues=', int(mean_queue_length))
    return mean_queue_length, queue_length, Delay, queue_list, delay_list

def get_phase(control_model, Phase, phase, lost_time, V_pass, V_SFR, lost_time_dic,n_weight, rand_vector,n_r):  
    # 该对应 non-cyclic BP，
    
    if control_model == 'BP': # Original BP
        q_num = copy.deepcopy(empty1)
        v_num = copy.deepcopy(empty1)
        weight = copy.deepcopy(empty1)
        for index in ['N','E']:
            for Dir in ['T']:
                if np.random.rand()<=success_rate and predict[index][Dir]==1:
                    is_pre = 1
                else:
                    is_pre = 0
                q_num[index][Dir],_ = Q_predict(is_pre,delta_t,lost_time_dic[index][Dir],V_SFR[index][Dir],h_mean[index][Dir],is_supplement_MSFR);     
                v_num[index][Dir] = len(Queue[index][Dir])
                if havetail == 1:
                    v_num[index][Dir] -= last_pass[index][Dir]
                weight[index][Dir] = q_num[index][Dir] * v_num[index][Dir]
        # 控制
        weight_list = []
        for i in range(len(Phase)):
            temp = 0
            for j in Phase[i]:
                index, Dir = j2Dir(j)
                temp += weight[index][Dir]
            weight_list.append(temp)
        index_temp = [i for i, j in enumerate(weight_list) if j == max(weight_list)] #返回weight_list中最大值的所有位置
        phase_new = index_temp[math.floor(rand_vector[n_r]*len(index_temp))] #np.random.choice(index_temp)
        n_r += 1;
        if n_r >= len(rand_vector):
            n_r = 0  #随机数读满了，再来一次
            
    elif control_model == 'FT': # fixed-time control
        if phase == 1: #3:
            phase_new = 0 
        else:
            phase_new = phase + 1

    elif control_model =='PWBP': # P BP
        q_num = copy.deepcopy(empty1)
        v_num = copy.deepcopy(empty1)
        weight = copy.deepcopy(empty1)
        _,queue_future = get_Queue(Arrival, Queue, T_step+delta_t)
        for index in ['N','E']:
            for Dir in ['T']:
                if np.random.rand()<=success_rate and predict[index][Dir]==1: # 有预测误差，该判断才实际生效
                    is_pre = 1
                else:
                    is_pre = 0
                q_num[index][Dir],_ = Q_predict(is_pre,delta_t,lost_time_dic[index][Dir],V_SFR[index][Dir],h_mean[index][Dir],is_supplement_MSFR)
                v_num[index][Dir] = get_weight(Queue[index][Dir], T_step, delta_t, n_weight)
                weight[index][Dir] = q_num[index][Dir] * v_num[index][Dir]
        # 控制
        weight_list = []
        for i in range(len(Phase)):
            temp = 0
            for j in Phase[i]:
                index, Dir = j2Dir(j)
                temp += weight[index][Dir]
            weight_list.append(temp)
        index_temp = [i for i, j in enumerate(weight_list) if j == max(weight_list)] #返回weight_list中最大值的所有位置
        phase_new = index_temp[math.floor(rand_vector[n_r]*len(index_temp))] #np.random.choice(index_temp)
        n_r += 1;
        if n_r >= len(rand_vector):
            n_r = 0  #随机数读满了，再来一次  
            
    elif control_model == 'LESCBP': # S BP
        q_num = copy.deepcopy(empty1)
        v_num = copy.deepcopy(empty1)
        weight = copy.deepcopy(empty1)
        _,queue_future = get_Queue(Arrival, Queue, T_step+delta_t)
        queue_sum = 0
        psi_a = 0
        psi_b = 0
        
        for index in ['N','E']:
            for Dir in ['T']:
                if np.random.rand()<=success_rate and predict[index][Dir]==1:
                    is_pre = 1
                else:
                    is_pre = 0
                q_num[index][Dir],_ = Q_predict(is_pre,delta_t,lost_time_dic[index][Dir],V_SFR[index][Dir],h_mean[index][Dir],is_supplement_MSFR);
                v_num[index][Dir] = get_weight(Queue[index][Dir], T_step, delta_t, n_weight)
                weight[index][Dir] = q_num[index][Dir] * v_num[index][Dir]
                queue_sum += len(queue_future[index][Dir])
        for i in range(len(Phase)):
            psi_a_temp = 0
            for j in Phase[i]:
                index_temp, Dir_temp = j2Dir(j)
                psi_a_temp += weight[index_temp][Dir_temp]
            psi_a = max(psi_a,psi_a_temp)
        for j in Phase[phase]:
            index_temp, Dir_temp = j2Dir(j)
            psi_b += weight[index_temp][Dir_temp]
        psi_c = 0.05 * queue_sum ** 0.1  #  alpha = 0.05, beta = 0.1
        # 控制
        if psi_a - psi_b - psi_c >0:
            weight_list = []
            for i in range(len(Phase)):
                temp = 0
                for j in Phase[i]:
                    index, Dir = j2Dir(j)
                    temp += weight[index][Dir]
                weight_list.append(temp)
            index_temp = [i for i, j in enumerate(weight_list) if j == max(weight_list)] #返回weight_list中最大值的所有位置
            phase_new = index_temp[math.floor(rand_vector[n_r]*len(index_temp))] #np.random.choice(index_temp)
            n_r += 1;
            if n_r >= len(rand_vector):
                n_r = 0  #随机数读满了，再来一次   
        else:
            phase_new = phase

    elif control_model == 'DelayBP':  # D BP
        q_num = copy.deepcopy(empty1)
        Delay_num = copy.deepcopy(empty1)
        weight = copy.deepcopy(empty1)
        for index in ['N','E']:
            for Dir in ['T']:
                if np.random.rand()<=success_rate and predict[index][Dir]==1:
                    is_pre = 1
                else:
                    is_pre = 0
                q_num[index][Dir],_ = Q_predict(is_pre,delta_t,lost_time_dic[index][Dir],V_SFR[index][Dir],h_mean[index][Dir],is_supplement_MSFR);
                Delay_num[index][Dir] = len(Queue[index][Dir]['Arrival_time'])*T_step - sum(Queue[index][Dir]['Arrival_time'].apply(lambda x: T_step if x >T_step else x)) # TTff为delta_t
                weight[index][Dir] = q_num[index][Dir] * Delay_num[index][Dir]
    
        # 控制
        weight_list = []
        for i in range(len(Phase)):
            temp = 0
            for j in Phase[i]:
                index, Dir = j2Dir(j)
                temp += weight[index][Dir]
            weight_list.append(temp)
        index_temp = [i for i, j in enumerate(weight_list) if j == max(weight_list)] #返回weight_list中最大值的所有位置
        phase_new = index_temp[math.floor(rand_vector[n_r]*len(index_temp))] #np.random.choice(index_temp)
        n_r += 1;
        if n_r >= len(rand_vector):
            n_r = 0  #随机数读满了，再来一次
    
    # 连续流福利--没有启动损失。但如果对应定周期，则不需要更新，因为一定有损失时间
    lost_time_dic = copy.deepcopy(empty_lost)
    if control_model != 'FixedBP' and control_model != 'FT' and control_model != 'TTBP':      
        for j in Phase[phase_new]:
            index,Dir = j2Dir(j)
            lost_time_dic[index][Dir] = 0
    
    return phase_new,lost_time_dic,n_r

def get_predict(predict_percent):
    if predict_percent == 'None':
        predict_percent = None
    elif predict_percent == 'All':
        predict_percent = 'NE'
        
    if predict_percent == None: # 研究常用方法，即，使用M-SFR信息
        predict = {'N':{'L':0,'T':0},  # 1 if predict, 0 otherwise (use mean value)
                   'S':{'L':0,'T':0},
                   'W':{'L':0,'T':0},
                   'E':{'L':0,'T':0}}
    elif predict_percent == 'N':
        predict = {'N':{'L':1,'T':1},
                   'S':{'L':0,'T':0},
                   'W':{'L':0,'T':0},
                   'E':{'L':0,'T':0}}
    elif predict_percent == 'W':
        predict = {'N':{'L':0,'T':0},
                   'S':{'L':0,'T':0},
                   'W':{'L':1,'T':1},
                   'E':{'L':0,'T':0}}
    elif predict_percent == 'S':
        predict = {'N':{'L':0,'T':0},
                   'S':{'L':1,'T':1},
                   'W':{'L':0,'T':0},
                   'E':{'L':0,'T':0}}
    elif predict_percent == 'E':
        predict = {'N':{'L':0,'T':0},
                   'S':{'L':0,'T':0},
                   'W':{'L':0,'T':0},
                   'E':{'L':1,'T':1}}
    elif predict_percent == 'NW':
        predict = {'N':{'L':1,'T':1},
                   'S':{'L':0,'T':0},
                   'W':{'L':1,'T':1},
                   'E':{'L':0,'T':0}}
    elif predict_percent == 'NS':
        predict = {'N':{'L':1,'T':1},
                   'S':{'L':1,'T':1},
                   'W':{'L':0,'T':0},
                   'E':{'L':0,'T':0}}
    elif predict_percent == 'NE': # 研究常用方法，即，使用I-SFR信息
        predict = {'N':{'L':1,'T':1},
                   'S':{'L':0,'T':0},
                   'W':{'L':0,'T':0},
                   'E':{'L':1,'T':1}}
    elif predict_percent == 'WS':
        predict = {'N':{'L':0,'T':0},
                   'S':{'L':1,'T':1},
                   'W':{'L':1,'T':1},
                   'E':{'L':0,'T':0}}
    elif predict_percent == 'WE':
        predict = {'N':{'L':0,'T':0},
                   'S':{'L':0,'T':0},
                   'W':{'L':1,'T':1},
                   'E':{'L':1,'T':1}}
    elif predict_percent == 'SE':
        predict = {'N':{'L':0,'T':0},
                   'S':{'L':1,'T':1},
                   'W':{'L':0,'T':0},
                   'E':{'L':1,'T':1}}
    elif predict_percent == 'NWE':
        predict = {'N':{'L':1,'T':1},
                   'S':{'L':0,'T':0},
                   'W':{'L':1,'T':1},
                   'E':{'L':1,'T':1}}
    elif predict_percent == 'NWS':
        predict = {'N':{'L':1,'T':1},
                   'S':{'L':1,'T':1},
                   'W':{'L':1,'T':1},
                   'E':{'L':0,'T':0}}
    elif predict_percent == 'WSE':
        predict = {'N':{'L':0,'T':0},
                   'S':{'L':1,'T':1},
                   'W':{'L':1,'T':1},
                   'E':{'L':1,'T':1}}
    elif predict_percent == 'NSE':
        predict = {'N':{'L':1,'T':1},
                   'S':{'L':1,'T':1},
                   'W':{'L':0,'T':0},
                   'E':{'L':1,'T':1}}
    elif predict_percent == 'NWSE':
        predict = {'N':{'L':1,'T':1},
                   'S':{'L':1,'T':1},
                   'W':{'L':1,'T':1},
                   'E':{'L':1,'T':1}}
    return predict

def get_Queue(Arrival_left, Queue, t_step):
    for index in ['N','E']:
        for Dir in ['T']:
            temp = Arrival_left[index][Dir]
            arrival_temp = temp[(temp['Arrival_time'] >= t_step) & (temp['Arrival_time'] < t_step + delta_t )]
            temp = temp.drop(temp[(temp['Arrival_time'] >= t_step) & (temp['Arrival_time'] < t_step + delta_t )].index)
            Arrival_left[index][Dir] = copy.deepcopy(temp)
            if len(arrival_temp) > 0:
                for i in arrival_temp.index:
                    Queue[index][Dir].loc[len(Queue[index][Dir])] = arrival_temp.loc[i] # add new arrival to queue
            Queue[index][Dir] = Queue[index][Dir].reset_index(drop=True)
    return Arrival_left, Queue

def get_weight(QueueFrame,T_step, delta_t,n_weight):
    weight_sum = 0
    pos_wei_now = 1
    for i_q in range(len(QueueFrame)):
        pos_wei_now = max(0,min(pos_wei_now,(T_step+delta_t - QueueFrame['Arrival_time'][i_q]) / delta_t))
        if pos_wei_now == 0:
            break
        else:
            weight_sum += pos_wei_now
        pos_wei_now -= 1/n_weight
    return weight_sum

def H_mean(Headway): # 计算平均车头时距
    h_mean = copy.deepcopy(empty1)
    SFR_mean = copy.deepcopy(empty1)
    h_std = copy.deepcopy(empty1)
    for index in ['N','E']:
        for Dir in ['T']:
            h_mean[index][Dir] = np.mean(Headway[index][Dir]['headway']/Headway[index][Dir]['veh_type'])
            h_std[index][Dir] = np.std(Headway[index][Dir]['headway']/Headway[index][Dir]['veh_type'])
            SFR_mean[index][Dir] = 3600 / np.mean(Headway[index][Dir]['headway']/Headway[index][Dir]['veh_type'])
    return h_mean, SFR_mean,h_std

def Headway_combine(Headway_origin, Lanes,type_consideration):
    Headway = copy.deepcopy(empty1)
    for index in ['N','E']:
        for Dir in ['T']:
            if Dir == 'L':
                Headway[index][Dir] = copy.deepcopy(Headway_origin[index][Dir])
            else:
                Headway[index][Dir] = pd.concat([Headway_origin[index]['T1'],Headway_origin[index]['T2']],axis=0)
                Headway[index][Dir] = Headway[index][Dir].reset_index(drop=True)
            Headway[index][Dir].columns = ['headway','veh_type']
            if type_consideration == 1:
                Headway[index][Dir]['headway'] = Headway[index][Dir]['headway']  / Lanes[index][Dir]/ Headway[index][Dir]['veh_type']
            elif type_consideration == 0:
                Headway[index][Dir]['headway'] = Headway[index][Dir]['headway']  / Lanes[index][Dir] # / Headway[index][Dir]['veh_type']
            Headway[index][Dir]['veh_type'] = Headway[index][Dir]['veh_type'] / Headway[index][Dir]['veh_type']
    return Headway

def initialize():
    # 初始化一些常用的字典，避免反复生成
    
    empty_lost = {'N':{'T': lost_time},
              'E':{'T': lost_time}}
    
    empty_delta_t = {'N':{'T': delta_t},
              'E':{'T': delta_t}}
    empty0 = {'N':{'T': 0},
              'E':{'T': 0}}
    
    
    empty1 = {'N':{'T':[]},
           'E':{'T':[]}}
      
    '''
    empty1 = {'N':{'L':[],'T':[]},
              'S':{'L':[],'T':[]},
              'W':{'L':[],'T':[]},
              'E':{'L':[],'T':[]}}
    '''
    
    col_name2 = ['Arrival_time','Arrival_interval','Veh_type']
    '''
    empty2 = {'N':{'L':pd.DataFrame(columns=col_name2),'T':pd.DataFrame(columns=col_name2)},
              'S':{'L':pd.DataFrame(columns=col_name2),'T':pd.DataFrame(columns=col_name2)},
              'W':{'L':pd.DataFrame(columns=col_name2),'T':pd.DataFrame(columns=col_name2)},
              'E':{'L':pd.DataFrame(columns=col_name2),'T':pd.DataFrame(columns=col_name2)}}
    '''
    empty2 = {'N':{'T':pd.DataFrame(columns=col_name2)},'E':{'T':pd.DataFrame(columns=col_name2)}}
    col_name3 = ['Arrival_time','Arrival_interval','Veh_type','Pass_headway','Pass_time','Delay']
    '''
    empty3 = {'N':{'L':pd.DataFrame(columns=col_name3),'T':pd.DataFrame(columns=col_name3)},
              'S':{'L':pd.DataFrame(columns=col_name3),'T':pd.DataFrame(columns=col_name3)},
              'W':{'L':pd.DataFrame(columns=col_name3),'T':pd.DataFrame(columns=col_name3)},
              'E':{'L':pd.DataFrame(columns=col_name3),'T':pd.DataFrame(columns=col_name3)}}
    '''
    empty3 = {'N':{'T':pd.DataFrame(columns=col_name3)},'E':{'T':pd.DataFrame(columns=col_name3)}}
    col_name4 = ['Time','Mean_Delay']
    '''
    empty4 = {'N':{'L':pd.DataFrame(columns=col_name4),'T':pd.DataFrame(columns=col_name4)},
              'S':{'L':pd.DataFrame(columns=col_name4),'T':pd.DataFrame(columns=col_name4)},
              'W':{'L':pd.DataFrame(columns=col_name4),'T':pd.DataFrame(columns=col_name4)},
              'E':{'L':pd.DataFrame(columns=col_name4),'T':pd.DataFrame(columns=col_name4)}}
    '''
    empty4 = {'N':{'T':pd.DataFrame(columns=col_name4)},'E':{'T':pd.DataFrame(columns=col_name4)}}
    Phase = [[2],[8]]
    # Phase = [[1,2],[3,4],[5,6],[7,8],[1,5],[2,6],[3,7],[4,8]]
    Lanes = {'N':{'T':1},'E':{'T':1}}
    Queue = copy.deepcopy(empty2)
    Departure = copy.deepcopy(empty3)
    Delay = copy.deepcopy(empty4)
    return empty_lost, empty_delta_t, empty0, empty1, empty2, empty3, Phase, Lanes, Queue, Departure, Delay

def j2Dir(j): # 流向转方向
    if j == 1:
        index = 'N'; Dir = 'L'
    elif j == 2:   # 研究的流向之一
        index = 'N'; Dir = 'T'
    elif j == 3:
        index = 'W'; Dir = 'L'
    elif j == 4:
        index = 'W'; Dir = 'T'
    elif j == 5:
        index = 'S'; Dir = 'L'
    elif j == 6:
        index = 'S'; Dir = 'T'
    elif j == 7:
        index = 'E'; Dir = 'L'
    elif j == 8:   # 研究的流向之一
        index = 'E'; Dir = 'T'
    return index, Dir

def Pass_time_gen(Headway,Queue,continuous_time_left,lost_time_dic,rand_vector,n_r):
    #输出即时的passing headway
    pass_time = copy.deepcopy(empty1) # 真实的通过时间， 针对 s^*(t) 和 q(t)
    pass_time_biased = copy.deepcopy(empty1) # 忽略了启动损失的通过时间， 针对 s(t)
    V_pass = copy.deepcopy(empty1)
    V_pass_biased = copy.deepcopy(empty1)
    for index in ['N','E']:
        for Dir in ['T']:
            for k in range(len(Queue[index][Dir])):  #几辆车在排队
                if k == 0 and isinstance(continuous_time_left[index][Dir],float): # it is first vehicle and Yes continuous
                    pass_time[index][Dir].append(continuous_time_left[index][Dir])  
                    pass_time_biased[index][Dir].append(continuous_time_left[index][Dir])
                else:
                    headway_sample = Headway[index][Dir].loc[int(rand_vector[n_r]*len(Headway[index][Dir]))];
                    n_r += 1;
                    if n_r >= len(rand_vector): # 随机数生成，用来debug,用处不大
                        n_r = 0  #随机数读满了，再来一次
                    pass_time[index][Dir].append(headway_sample['headway']/headway_sample['veh_type']*Queue[index][Dir].loc[k,'Veh_type']); # 大车转小车
                    pass_time_biased[index][Dir].append(headway_sample['headway']/headway_sample['veh_type']*Queue[index][Dir].loc[k,'Veh_type']);
                    if k == 0 and lost_time_dic[index][Dir]>0: # 有损失
                         pass_time[index][Dir][0] += lost_time_dic[index][Dir]
            
            # 开始衡量实际通过
            t_mark = 0
            k_veh = 0
            while t_mark < 2 * delta_t and k_veh<len(pass_time[index][Dir]):
                t_mark_new = pass_time[index][Dir][k_veh]+max(t_mark,Queue[index][Dir]['Arrival_time'][k_veh]-T_step)
                V_pass[index][Dir].append(t_mark_new-t_mark)
                t_mark = t_mark_new
                k_veh +=1
            
            # 开始不考虑启动损失的理想通过
            t_mark = 0
            k_veh = 0
            while t_mark < 2 * delta_t and k_veh<len(pass_time_biased[index][Dir]):
                t_mark_new = pass_time_biased[index][Dir][k_veh]+max(t_mark,Queue[index][Dir]['Arrival_time'][k_veh]-T_step)
                V_pass_biased[index][Dir].append(t_mark_new-t_mark)
                t_mark = t_mark_new
                k_veh +=1            
            
            # 开始衡量饱和流率 
    V_SFR = copy.deepcopy(pass_time)
    V_SFR_biased = copy.deepcopy(pass_time_biased)
    for index in ['N','E']:
        for Dir in ['T']:
            k = len(V_SFR[index][Dir])
            while sum(V_SFR[index][Dir])<delta_t:
                V_SFR[index][Dir].append(h_mean[index][Dir])
                k += 1

            # 开始衡量有偏的饱和流率 
    for index in ['N','E']:
        for Dir in ['T']:
            k = len(V_SFR_biased[index][Dir])
            while sum(V_SFR_biased[index][Dir])<delta_t:
                V_SFR_biased[index][Dir].append(h_mean[index][Dir])
                k += 1

    return V_pass,V_SFR,V_pass_biased,V_SFR_biased,n_r


def Q_predict(is_predict,t_discharge,lost_time_part,pass_list,Mheadway,is_supplement_MSFR):
    # 注意 如何输入的是V_SFR,则预测的是饱和流率。如果输入的是V_pass,则是计算通过量
    # 简化法算通过了几辆车   
    time_left = []  # 需要二次通过的车辆，如果能连续通过，还要多久时间。只在实际放行时，才可能被记录
    if is_predict == 0: #if do not predict I-SFR 
        if is_true_pass == 1:
            n_pass = (t_discharge-lost_time_part)/Mheadway 
        elif is_true_pass == 0:
            n_pass = t_discharge/Mheadway
    else:  #if predict I-SFR # 损失时间的部分被考虑在了 pass_list 里面 如果是 it_true_pass == 0, 输入的其实是 V_SFR_biased，即不考虑 lost time 部分
        cumulative_time = 0
        if len(pass_list) > 0:
            for pass_index in range(len(pass_list)):
                cumulative_time += pass_list[pass_index]
                if cumulative_time > t_discharge:
                    pass_index -= 1
                    time_left = cumulative_time-t_discharge
                    break
            n_pass = pass_index+1
        else:
            n_pass = 0
        if is_supplement_MSFR == 1 and cumulative_time<t_discharge:
            n_pass = n_pass + (t_discharge-cumulative_time)/Mheadway
    return n_pass,time_left

def Rand_generate(num,name):
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir+'\\rand\\', name)
    if os.path.isfile(file_path): # if the file exists, load it
        rand_vector = pickle.load(open('rand/'+name, "rb"))
    else: #generate new file otherwise
        rand_vector = np.random.rand(num)
        pickle.dump(rand_vector, open('rand/'+name, "wb"))
    return rand_vector

def update_Departure(phase, last_phase, V_pass, T_now,t_discharge, Departure, Queue): 
    i = 0
    last_pass = copy.deepcopy(empty0)
    continuous_time_left = copy.deepcopy(empty1)
    for j in Phase[phase]: 
        index, Dir = j2Dir(j)
        n_pass,time_left = Q_predict(1,t_discharge,_,V_pass[index][Dir],h_mean[index][Dir],0)
        continuous_time_left[index][Dir] = time_left
        last_pass[index][Dir] = n_pass
        cumulative_time = 0 
        i += 1
        for k in range(n_pass):
            cumulative_time += V_pass[index][Dir][k]
            pass_time = T_now + cumulative_time
            depart_temp = Queue[index][Dir].loc[k].tolist()
            if k==0 and len(Departure[index][Dir])>0 and j in Phase[last_phase]: # 获得真实headway
                headway_true = T_now+cumulative_time - max(Queue[index][Dir]['Arrival_time'][0],Departure[index][Dir]['Pass_time'][len(Departure[index][Dir])-1])
            else:
                headway_true = min(copy.deepcopy(V_pass[index][Dir][k]),T_now+ cumulative_time - Queue[index][Dir]['Arrival_time'][k])
            delay = pass_time - depart_temp[0] - headway_true
            depart_temp.extend([headway_true, pass_time, delay])
            Departure[index][Dir].loc[len(Departure[index][Dir])] = depart_temp
            Queue[index][Dir] = Queue[index][Dir].drop(Queue[index][Dir].index[0]) # (k)# 把Queue目前的第一行删除
        Queue[index][Dir] = Queue[index][Dir].reset_index(drop=True)
    return Departure, Queue, continuous_time_left

def weight_fixed_BP(control_model,V_pass,V_SFR,Phase,Departure, Queue,last_phase, n_weight):
    # 针对 cyclic BP 计算各个phase 长度，并控制 # cyclic BP 每个周期决策一次
    
    SFR_num = copy.deepcopy(empty1)
    v_num = copy.deepcopy(empty1)
    weight = [0]*len(Phase)
    eta = 2.5 
    loop_times = 5

    '''    
    for i_loop in range(loop_times):
        weight_old = copy.deepcopy(weight)
        weight = [0]*len(Phase)
        phase_split = [0]*len(Phase)
        percent_split  = copy.deepcopy(empty_delta_t)
        for index in ['N','E']:
            for Dir in ['T']:
                #add_lost = lost_time
                #for j in Phase[phase]:
                #    index0, Dir0 = j2Dir(j)
                #    if index==index0 and Dir==Dir0:
                #        add_lost = 0
                est_index = min(i_loop, predict[index][Dir]) # 第一次循环为0， 如果不预测，也为0
                if i_loop==0 or est_index==1: # 第一次循环，或需要I-SFR时，进入
                    if i_loop==0: # 算一次就够了
                        v_num[index][Dir] = len(V_pass[index][Dir])
                        # print(v_num)
                    # else:
                    #    phase_seq = dir2phase(index,Dir,Phase)
                    SFR_num[index][Dir],_ = Q_predict(est_index,percent_split[index][Dir],V_SFR[index][Dir],h_mean[index][Dir])
                    # print(SFR_num)
                    for phase_inc in dir2phase(index, Dir, Phase):
                        weight[phase_inc] += SFR_num[index][Dir] * v_num[index][Dir]
        sum_exp_weight = 0
        # print(weight)
        for i_w in range(len(Phase)):
            sum_exp_weight += math.exp(eta*weight[i_w])
        for i_w in range(len(Phase)):
            phase_split[i_w] = math.exp(eta*weight[i_w])/sum_exp_weight # 注意，他们15年的方法，1个movement在两个phases中，会被分别算在内 
            for j in Phase[i_w]:
                index, Dir = j2Dir(j)
                percent_split[index][Dir] += phase_split[i_w]
        if weight_old==weight:
            break
    '''
    if control_model == 'FixedBP':  # E BP
        weight = [0]*len(Phase)
        phase_split = [0]*len(Phase)
        percent_split  = copy.deepcopy(empty_delta_t)
        for index in ['N','E']:
            for Dir in ['T']:
                v_num[index][Dir] = len(Queue[index][Dir])
                if np.random.rand()<=success_rate and predict[index][Dir]==1:
                    is_pre = 1
                else:
                    is_pre = 0
                SFR_num[index][Dir],_ = Q_predict(is_pre,percent_split[index][Dir],lost_time_dic[index][Dir],V_SFR[index][Dir],h_mean[index][Dir],is_supplement_MSFR)
                for phase_inc in dir2phase(index, Dir, Phase):
                    weight[phase_inc] += eta* SFR_num[index][Dir] * v_num[index][Dir]
                    
        weight_max = max(weight)
        for i_w in range(len(weight)):
            weight[i_w] = weight[i_w] - weight_max
        sum_exp_weight = 0
        for i_w in range(len(Phase)):
            sum_exp_weight += math.exp(weight[i_w])
        for i_w in range(len(Phase)):
            phase_split[i_w] = math.exp(weight[i_w])/sum_exp_weight # 注意，他们15年的方法，1个movement在两个phases中，会被分别算在内，但对本研究无影响
        T_now = T_step
        for phase in phase_sequence:
            Departure, Queue, _ = update_Departure(phase,last_phase, V_pass, T_now, delta_t*phase_split[phase], Departure, Queue)
            T_now += delta_t*phase_split[phase]
            continuous_time_left = copy.deepcopy(empty1)
            last_phase = phase
    
    elif control_model == 'TTBP':  # T BP
        weight = [0]*len(Phase)
        phase_split = [0]*len(Phase)
        percent_split  = copy.deepcopy(empty_delta_t)
        for index in ['N','E']:
            for Dir in ['T']:
                if np.random.rand()<=success_rate and predict[index][Dir]==1:
                    is_pre = 1
                else:
                    is_pre = 0
                v_num[index][Dir] = 58.93 * min(1,len(Queue[index][Dir])/n_weight)**3.59 + 6.17
                SFR_num[index][Dir],_ = Q_predict(is_pre,percent_split[index][Dir],lost_time_dic[index][Dir],V_SFR[index][Dir],h_mean[index][Dir],is_supplement_MSFR)
                for phase_inc in dir2phase(index, Dir, Phase):
                    weight[phase_inc] +=   SFR_num[index][Dir] * v_num[index][Dir]
        
        green_eff_min = 2.5
        for i_w in range(len(Phase)):
            phase_split[i_w] = weight[i_w]/sum(weight)  
        T_now = T_step
        for phase in phase_sequence: # 注意，如果有双环结构，这个写法不对，要为双环设置好开始和结束时间，但对本研究无影响
            delta_split = (delta_t-2*green_eff_min) * phase_split[phase]+green_eff_min
            Departure, Queue, _ = update_Departure(phase,last_phase, V_pass, T_now, delta_t*phase_split[phase], Departure, Queue)
            T_now += delta_split
            continuous_time_left = copy.deepcopy(empty1)
            last_phase = phase
        
    
    return Departure, Queue, continuous_time_left
    
    
current_time = datetime.datetime.now().strftime("%H_%M_%S_%f")
Headway_origin = pickle.load(open('Headway_origin.p', "rb"))
#####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 这是主控制模块，几乎所有控制参数都集成在这里
control_model = 'BP' # 可选 'DelayBP' 'LESCBP' 'PWBP' 'BP' 'FT' 'FixedBP' 'TTBP' 
# 'DelayBP' 对应 D, # 'LESCBP' 对应 S, # 'PWBP' 对应 P # 'BP' 对应 Original # 'FixedBP' 对应 E, # 'TTBP' 对应 T
control_name = 'Nf' #'Nf' # 'Af' # 'N' # 'A' # 'Q'
# 五个分别对应 \overline{s}, s(t), \overline{s}^*, s^*(t) 和 q(t)
T_long = 3600*24*3
warm_up_time = 3600
delta_t = 10
statistic_interval = delta_t # 3600 * 6 # 600
lost_time = 2.5
total_spl = 100 ; perc_low = 50 ; perc_high = 51 #total_spl+1 ;   # 50
success_rate = 1 # infor = 'All' 时生效
is_origin = 2  # 1 是真实数据，0 是方差缩小数据， 2 是方差放大数据 # -1 无方差
havetail = 0  # 是否考虑下游权重，0为不考虑，1为考虑离开的一个Delta_t时间的车辆。# 因为影响较小，意义也不大，所以没在论文中讨论
n_weight = 20
if control_model=='FixedBP' or control_model=='TTBP' or control_model=='FT':
    delta_t = 90 # 这里delta_t 是定时控制的周期时长 
    phase_sequence = [0,1]
    if control_model == 'FT':
        delta_t = delta_t/2
    statistic_interval = max(delta_t,statistic_interval) # 判断稳定与否的统计间隔
##############################################################################
is_Q = 0
if control_name == 'N' or control_name == 'None' or control_name == 'None_true':
    infor = 'None'
    is_true_pass = 1 # 在计算饱和流率或通过量预测时，=1 表示phase计算时切换的损失直接扣除； =0 表示不phase计算不考虑损失。
elif control_name == 'Nf' or control_name == 'None_biased':
    infor = 'None'
    is_true_pass = 0 # 在计算饱和流率或通过量预测时，=1 表示phase计算时切换的损失直接扣除； =0 表示不phase计算不考虑损失。
elif control_name == 'A' or control_name == 'All' or control_name == 'All_true':
    infor = 'All'
    is_true_pass = 1
elif control_name == 'Af' or control_name == 'All_biased':
    infor = 'All'
    is_true_pass = 0
elif control_name == 'Q':
    infor = 'All'
    is_Q = 1
    is_true_pass = 1



if infor == 'None' or infor == None:
    success_rate = 0
    
type_consideration = 1 # 如果 是 1 就把大车headway 数据除以2 合并入小车数据 
method = 3 # 稳定判断方法

if type_consideration == 0 and is_origin == 1: # 根据经验 设置不同的搜索范围起始点 # 当搜索范围较大，不影响；当搜索范围较小，需合理设置
    S_N1 = 1305 # 1374  # N 轴的起点
    S_N2 = 0
    S_E1 = 1441 # 1532 # E 轴的起点
    S_E2 = 0

else:
    S_N1 = 1374
    S_N2 = 0
    S_E1 = 1532
    S_E2 = 0
    if control_model == 'FixedBP':
        S_N1 = 1374-50 # 1374
        S_N2 = 0
        S_E1 = 1532 -50 # 1532
        S_E2 = 0
    if control_model == 'TTBP':
        S_N1 = 1374-100 # 1374
        S_N2 = 0
        S_E1 = 1532 -100 # 1532
        S_E2 = 0
    if control_model == 'FT':
        S_N1 = 1374/2
        S_N2 = 0
        S_E1 = 1532/2
        S_E2 = 0        
        
random_seed = 10
is_supplement_MSFR = 1 
new_frontier = []
arrival_name = 'Arrival'+str(random_seed)+'_add'
rand_name = 'rand_vector_'+str(random_seed)
stationary_dict = {}
rand_vector = Rand_generate(500000,rand_name+'.p')
n_r = 0; # random number choice # 该参数用于debug中控制随机性，保留不影响，可以忽略
last_phase = 0
## for debug
#col_namex = ['n-pass','is_success']
#npass_record = {'N':{'T':pd.DataFrame(columns=col_namex)},'E':{'T':pd.DataFrame(columns=col_namex)}}

'''
if is_origin != 1:
    headway_dataE = np.random.exponential(2.345, 10000)
    headway_dataN = np.random.exponential(2.620, 10000)
    veh_type_data = [1] * 10000
    dataE = {'headway': headway_dataE, 'veh_type': veh_type_data}
    dataN = {'headway': headway_dataN, 'veh_type': veh_type_data}
    Headway = {'E':{'T':pd.DataFrame(dataE)},'N':{'T':pd.DataFrame(dataN)}}
'''
empty1 = {'N':{'T':[]},'E':{'T':[]}}
Headway = Headway_combine(Headway_origin,{'N':{'T':1},'E':{'T':1}},type_consideration)
if is_origin == 0:
    for index in ['N','E']:
        for Dir in ['T']:
            mean_value = Headway[index][Dir]['headway'].mean()
            Headway[index][Dir]['headway'] = (Headway[index][Dir]['headway']+mean_value)/2
elif is_origin == 2:
    for index in ['N','E']:
        for Dir in ['T']:
            headway_ori = copy.deepcopy(Headway[index][Dir])
            for times_idx in [0.5,1.5]:
                headway_add = copy.deepcopy(headway_ori)
                headway_add['headway'] = headway_add['headway'] * times_idx
                Headway[index][Dir] = Headway[index][Dir]._append(headway_add, ignore_index=True)            
elif is_origin ==-1:
    for index in ['N','E']:
        for Dir in ['T']:
            mean_value = Headway[index][Dir]['headway'].mean()
            Headway[index][Dir]['headway'] = Headway[index][Dir]['headway']*0+mean_value

for predict_percent in [infor]: # 'None', 'All'
    n_start = 0
    try:
        del new_frontier
    except NameError:
        pass
    
    #-----------------------------------------------------------------
    for perc in range(perc_low,perc_high):
        print('perc is '+str(perc))
        g_N =  perc/total_spl #0.5 
        g_E = 1-g_N
        D0 = {'N':{'T1':S_N1*g_N,'T2':S_N2*g_N},'E':{'T1':S_E1*g_E,'T2':S_E2*g_E}}
        add_rand =  np.random.rand()
        add_demand_s = -392+add_rand*100 
        add_demand_ns = 120+add_rand*100 
        if control_model == 'FT':
            add_demand_s = -424+add_rand*100 
            add_demand_ns = 600+add_rand*100 
        count_i = 0
        while np.abs(add_demand_ns-add_demand_s) > 2.00: # 400: # 稳定与不稳定的add_demand还未收敛
            add_demand = (add_demand_s + add_demand_ns) / 2
            add_demand = int(add_demand)
            empty_lost, empty_delta_t, empty0, empty1, empty2, empty3, Phase, Lanes, Queue, Departure, Delay = initialize()
            h_mean, SFR_mean, h_std = H_mean(Headway)
            D = flow_add_change(D0, add_demand)
            ds_ratio = ds_ratio_estimate(SFR_mean, delta_t, lost_time, D)
            print(ds_ratio, 'predict_percent=', predict_percent, 'add_demand=', add_demand, 'control_model=', control_model, 'Demand[N,E]=',str(D['N']['T']),str(D['E']['T']))
            Arrival = Arrival_generate('',Headway,delta_t,D,0,'')
            predict = get_predict(predict_percent)
            
            switch_frequency = 0
            total_frequency = 0
            
            phase = 0
            queue_length = 0
            queue_list = [] 
            delay_list = []
            continuous_time_left = copy.deepcopy(empty1)
            lost_time_dic =copy.deepcopy(empty_lost)
            last_pass = copy.deepcopy(empty0)
            SFR_N = []
            SFR_E = []
            
            for i_time in range(int(T_long/(delta_t))): # range(int(T/(delta_t+lost_time))):
                T_step = i_time * delta_t 
                Arrival = Arrival_generate('',Headway,T_step+2*delta_t ,D ,0.5, Arrival)
                Arrival, Queue = get_Queue(Arrival, Queue, T_step) #获取新到的排队车辆
                V_pass,V_SFR,_,V_SFR_biased,n_r = Pass_time_gen(Headway,Queue,continuous_time_left,lost_time_dic,rand_vector,n_r); # 获取随机的ISFR
                if control_model!='FixedBP' and control_model!='TTBP':
                    last_phase = phase # 只用作headway 修正，对于FixedBP，last phase 始终为-1
                    if is_Q == 1:
                        phase, lost_time_dic, n_r = get_phase(control_model, Phase, phase, lost_time, V_pass, V_pass,lost_time_dic,n_weight,rand_vector,n_r) # 确定信号控制方案 # 2nd V_pass should be V_SFR
                    else:
                        if is_true_pass ==1:
                            phase, lost_time_dic, n_r = get_phase(control_model, Phase, phase, lost_time, V_pass, V_SFR,lost_time_dic,n_weight,rand_vector,n_r) # 确定信号控制方案 # 2nd V_pass should be V_SFR
                        elif is_true_pass ==0:
                            phase, lost_time_dic, n_r = get_phase(control_model, Phase, phase, lost_time, V_pass, V_SFR_biased,lost_time_dic,n_weight,rand_vector,n_r)
                    Departure, Queue, continuous_time_left = update_Departure(phase, last_phase, V_pass, T_step, delta_t, Departure, Queue) # 对每一个放行流向，更新排队
                    if phase!= last_phase:
                        switch_frequency += 1
                    total_frequency +=1
                    
                    if Phase[phase][0] == 2: # N
                        index = 'N'; Dir = 'T';
                        real_SFR,_ = Q_predict(1,delta_t,_,V_SFR_biased[index][Dir],h_mean[index][Dir],is_supplement_MSFR);     
                        SFR_N.append(real_SFR)
                        
                    elif Phase[phase][0] == 8: # E
                        index = 'E'; Dir = 'T';
                        real_SFR,_ = Q_predict(1,delta_t,_,V_SFR_biased[index][Dir],h_mean[index][Dir],is_supplement_MSFR);     
                        SFR_E.append(real_SFR)
                    
                else: # for the 'FixedBP' and 'TTBP'
                    if is_Q == 1:
                        Departure, Queue, continuous_time_left = weight_fixed_BP(control_model,V_pass,V_pass,Phase,Departure, Queue,last_phase,n_weight) # 2nd V_pass should be V_SFR
                    else:
                        if is_true_pass == 1:
                            Departure, Queue, continuous_time_left = weight_fixed_BP(control_model,V_pass,V_SFR,Phase,Departure, Queue,last_phase,n_weight) # 2nd V_pass should be V_SFR
                        elif is_true_pass == 0:
                            Departure, Queue, continuous_time_left = weight_fixed_BP(control_model,V_pass,V_SFR_biased,Phase,Departure, Queue,last_phase,n_weight) # 2nd V_pass should be V_SFR
                mean_queue_length, queue_length, Delay, queue_list, delay_list = get_delay(T_step, Queue, Departure, queue_length, Delay, queue_list, delay_list) # 统计延误
                if isinstance(mean_queue_length,float): 
                    # 可去除部分 明显稳定或不稳定 加速算法
                    if mean_queue_length > 640: 
                        stationary = 0
                        break
                    elif T_step>=3600*4 and mean_queue_length<3*(T_step/3600)**0.5 and T_step<=3600*24*7:
                        stationary = 1
                        if method == 3 and len(queue_list)>10:
                            z = [queue_list[i+1]-queue_list[i] for i in range(len(queue_list)-1)]
                            stat, p_value = ttest_1samp(z, 0,alternative="greater")
                            stationary = 0 if (p_value <= 0.05) else 1 # 原假设为 稳定，小于临界值，拒绝原假设
                        break
                    elif T_step>=3600*4 and T_step<=3600*24*7 and max(queue_list) <= (T_step/3600*40)**0.5:
                        stationary = 1
                        if method == 3 and len(queue_list)>10:
                            z = [queue_list[i+1]-queue_list[i] for i in range(len(queue_list)-1)]
                            stat, p_value = ttest_1samp(z, 0,alternative="greater")
                            stationary = 0 if (p_value <= 0.05) else 1 # 原假设为 稳定，小于临界值，拒绝原假设
                        break
            if i_time == int(T_long/(delta_t)) - 1: # 跑完了所有的时间，未提前终止仿真
                if method ==1:
                    dfgls1 = DFGLS(np.array(queue_list))
                    dfgls2 = DFGLS(delay_list)
                    stationary = 1 if (dfgls1.pvalue <= 0.05 or dfgls2.pvalue <= 0.05) else 0
                elif method == 2:
                    adf1 = adfuller(queue_list)
                    adf2 = adfuller(delay_list)                
                    stationary = 1 if (adf1[1] <= 0.05 or adf2[1] <= 0.05) else 0
                elif method == 3:
                    z = [queue_list[i+1]-queue_list[i] for i in range(len(queue_list)-1)]
                    stat, p_value = ttest_1samp(z, 0,alternative="greater")
                    stationary = 0 if (p_value <= 0.05) else 1 # 原假设为 稳定，小于临界值，拒绝原假设
                    
            z = [queue_list[i+1]-queue_list[i] for i in range(len(queue_list)-1)]
            stat, p_value = ttest_1samp(z, 0)
            print(stat,p_value)
            if stationary == 0: # 若不稳定,更新add_demand_ns
                add_demand_ns = add_demand
            else: #若稳定，则更新
                add_demand_s = add_demand    
                
            if n_start == 0:
                new_demand = np.array([D['N']['T'],D['E']['T'],stationary])
                n_start += 1
            else:
                new_row = np.array([D['N']['T'],D['E']['T'],stationary])
                new_demand = np.vstack((new_demand,new_row))
        D_final = flow_add_change(D0, (add_demand_ns+add_demand_s)/2)
        
        new_row = np.array([D_final['N']['T'],D_final['E']['T']])
        try:
            new_frontier = np.vstack((new_frontier,new_row))
        except NameError:
            new_frontier = np.array([D_final['N']['T'],D_final['E']['T']])
    stationary_dict[str(predict_percent)+'_all'] = new_demand
    stationary_dict[str(predict_percent)+'_ft'] = new_frontier
    
if not os.path.exists('stationary_dict'):
    os.makedirs('stationary_dict')


if total_frequency>0:
    print('isor: ',is_origin,' switch frequency: ', switch_frequency / total_frequency)

pickle.dump(stationary_dict, open('stationary_dict/'+control_model+'_'+control_name+'_'+str(delta_t)+'_'+str(lost_time)+'_isor_'+str(is_origin)+'_sr_'+str(success_rate)+'_tail_'+str(havetail)+'_time_'+str(current_time)+'.p', "wb")) # '_tpcsd_'+str(type_consideration)+

