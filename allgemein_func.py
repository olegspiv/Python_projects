# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 14:51:41 2020

@author: olegs
"""


import numpy as np
from numpy import *

from scipy.io import loadmat
data = loadmat('paco.mat')
# define relevant variables
TrialList = data['TrialList']
EyeX = data['EyeX']
EyeY = data['EyeY']
TargetX = data['TargetX']
TargetY = data['TargetY']
# find errors
where = TrialList[:,3]
find_false = []
for i in range(0,len(where)):
  if where[int(i)]== 0:
    find_false.append(i)
    
EyeX = np.delete(EyeX,find_false,1)
EyeY = np.delete(EyeY,find_false,1)
TargetX = np.delete(TargetX,find_false,1)
TargetY = np.delete(TargetY,find_false,1)   

def saccade(Eye_velocity, vel_thres, down_vel_thres, sooka, start, new_max_sac_size):

  if new_max_sac_size == 0:
    max_sac_size = 70

  else:
    max_sac_size = new_max_sac_size

  vector = []
  count = -1
  for index in sooka:
    count += 1
    vector.append(index)  
    if vector[count] >= vel_thres:
      break
  del index  

  if len(vector) == len(sooka):    
    saccade_status = 0
    saccade_onset = 0
    saccade_offset = 0
    peak_vel = 0
    saccade_duration = 0
  else:  
    saccade_status = 1
    # vector.reverse()
    vector_inv = vector[::-1]
      
    vector_2 = []
    count = -1
    for index in vector_inv:
      count += 1
      vector_2.append(index) 
      if vector_2[count] <= down_vel_thres:
        break

    del index

    if 'old_saccade_offset' in locals():
      saccade_onset = old_saccade_offset + len(vector) - len(vector_2)
    else:
      saccade_onset = start + len(vector)-len(vector_2)  

    find_offset = saccade_onset + max_sac_size
    
    if find_offset > len(Eye_velocity):
      sooka_2 = Eye_velocity[saccade_onset:len(Eye_velocity)] 
    else:
      sooka_2 = Eye_velocity[saccade_onset:find_offset+1]  

    # sooka_2.reverse()
    sooka_2_inv = sooka_2[::-1]
      
    vector_3 = []
    count = -1
    for index in sooka_2_inv:
      count += 1
      vector_3.append(index)
      if vector_3[count]>=vel_thres:
        break
    del index    

    # vector_3.reverce()
    vector_inv_2 = vector_3[::-1]
    
    vector_4 = []
    count = -1
    for index in vector_inv_2:
      count += 1
      vector_4.append(index) 
      if vector_4[count] <= down_vel_thres:
        break
    del index
        
    saccade_offset = saccade_onset + len(sooka_2) - len(vector_3) + len(vector_4)


    if saccade_offset <= len(Eye_velocity):
      peak_vel = max(Eye_velocity[saccade_onset:saccade_offset+1])
    else:
      peak_vel = max(Eye_velocity[saccade_onset:len(Eye_velocity)])

    saccade_duration = saccade_offset-saccade_onset

  return saccade_onset, saccade_offset, peak_vel, saccade_duration, saccade_status

def plotting1(EyeX, EyeY, Saccade, Eye_velocity, index, q, trial_num):
    
  import matplotlib.pyplot as plt  
  axes = plt.subplot(211)
  plt.plot(EyeX[:, index])
  plt.plot(EyeY[:, index], color='red')
  ymin, ymax = axes.get_ylim()
  
  if q == index:
    plt.axvline(Saccade['onset'][q], ymin, ymax, color='red')
    plt.axvline(Saccade['offset'][q], ymin, ymax, color='red')
    plt.axvline(Saccade['target_jump'][q], ymin, ymax, color='black')
    plt.title('block ' + str(Saccade['block_num'][q]) + ', trial ' + str(q) + ', target ' + str(Saccade['sac_targetX'][q]))
    plt.xlim(Saccade['target_jump'][q] - 100, Saccade['target_jump'][q] + 2000)

  else:
    plt.axvline(Saccade['onset'][q-trial_num], ymin, ymax, color='red')
    plt.axvline(Saccade['offset'][q-trial_num], ymin, ymax, color='red')
    plt.axvline(Saccade['target_jump'][q-trial_num], ymin, ymax, color='black')
    plt.title('block' + str(Saccade['block_num'][q-trial_num]) + ', trial' + str(q-trial_num) + ', target' + str(Saccade['sac_targetX'][q-trial_num]))
    plt.xlim(Saccade['target_jump'][q-trial_num] - 100, Saccade['target_jump'][q-trial_num] + 2000)  
 
    
  axes2 = plt.subplot(212)   
  plt.plot(Eye_velocity)
  ymin, ymax = axes2.get_ylim()

  if q == index:
    plt.axvline(Saccade['onset'][q], ymin, ymax, color='red')
    plt.axvline(Saccade['offset'][q], ymin, ymax, color='red')
    plt.axvline(Saccade['target_jump'][q], ymin, ymax, color='black')
    plt.xlim(Saccade['target_jump'][q] - 100, Saccade['target_jump'][q] + 2000)
    plt.show()
  else:
    plt.axvline(Saccade['onset'][q-trial_num], ymin, ymax, color='red')
    plt.axvline(Saccade['offset'][q-trial_num], ymin, ymax, color='red')
    plt.axvline(Saccade['target_jump'][q-trial_num], ymin, ymax, color='black')
    plt.xlim(Saccade['target_jump'][q-trial_num] - 100, Saccade['target_jump'][q-trial_num] + 2000)  
    plt.show()
    
def plotting2(EyeX, EyeY, Saccade, Eye_velocity, index, q, trial_num, saccade_onset, saccade_offset):
    
  import matplotlib.pyplot as plt  
  axes = plt.subplot(211)
  plt.plot(EyeX[:, index])
  plt.plot(EyeY[:, index], color='red')
  ymin, ymax = axes.get_ylim()
  
  if q == index:
    plt.axvline(Saccade['onset'][q], ymin, ymax, color='red')
    plt.axvline(Saccade['offset'][q], ymin, ymax, color='red')
    plt.axvline(Saccade['target_jump'][q], ymin, ymax, color='black')
    plt.title('block ' + str(Saccade['block_num'][q]) + ', trial ' + str(q) + ', target ' + str(Saccade['sac_targetX'][q]))
    plt.xlim(Saccade['target_jump'][q] - 100, Saccade['target_jump'][q] + 2000)
    plt.axvline(saccade_onset, ymin, ymax, color='green')
    plt.axvline(saccade_offset, ymin, ymax, color='green')
  else:
    plt.axvline(Saccade['onset'][q-trial_num], ymin, ymax, color='red')
    plt.axvline(Saccade['offset'][q-trial_num], ymin, ymax, color='red')
    plt.axvline(Saccade['target_jump'][q-trial_num], ymin, ymax, color='black')
    plt.title('block' + str(Saccade['block_num'][q-trial_num]) + ', trial' + str(q-trial_num) + ', target' + str(Saccade['sac_targetX'][q-trial_num]))
    plt.xlim(Saccade['target_jump'][q-trial_num] - 100, Saccade['target_jump'][q-trial_num] + 2000)  
    plt.axvline(saccade_onset, ymin, ymax, color='green')
    plt.axvline(saccade_offset, ymin, ymax, color='green') 
    
  axes2 = plt.subplot(212)   
  plt.plot(Eye_velocity)
  ymin, ymax = axes2.get_ylim()

  if q == index:
    plt.axvline(Saccade['onset'][q], ymin, ymax, color='red')
    plt.axvline(Saccade['offset'][q], ymin, ymax, color='red')
    plt.axvline(Saccade['target_jump'][q], ymin, ymax, color='black')
    plt.xlim(Saccade['target_jump'][q] - 100, Saccade['target_jump'][q] + 2000)
    plt.axvline(saccade_onset, ymin, ymax, color='green')
    plt.axvline(saccade_offset, ymin, ymax, color='green')     
    plt.show()
  else:
    plt.axvline(Saccade['onset'][q-trial_num], ymin, ymax, color='red')
    plt.axvline(Saccade['offset'][q-trial_num], ymin, ymax, color='red')
    plt.axvline(Saccade['target_jump'][q-trial_num], ymin, ymax, color='black')
    plt.xlim(Saccade['target_jump'][q-trial_num] - 100, Saccade['target_jump'][q-trial_num] + 2000)  
    plt.axvline(saccade_onset, ymin, ymax, color='green')
    plt.axvline(saccade_offset, ymin, ymax, color='green')     
    plt.show()    


da = 0 #input ('1 if dark adapted, 0 if not ')
la = 0  
which_date = 12102018


# velocity in degrees per second
vel_thres = 50
down_vel_thres = 5 

time_ms = []
for i in range(1,len(EyeX)):
  time_ms.append(i)
time_ms.append(len(time_ms)+1)

# time_ms = time_ms'

if da == 1:
  lum = 0 
else: 
  lum = 1


block_num = 1 #input ('input block number ');
tgt_size = 0.02 #input ('input target size in degrees ');


new_max_sac_size = 0

if 'Saccade' in locals():
  q = len(Saccade)
  p = q
else:
  Saccade = {'block_num':[], 'date':[], 'lum': [], 'onset': [], 'offset': [], 'peak_velocity': [], 'duration': [], 'size': [],'latency': [], 'init_targetX': [], 'sac_targetX': [], 'target_jump': [], 'target_ampl':[],'dark_adapt':[],'light_adapt':[], 'tgt_size':[]}
  

if 'p' not in locals(): 
  p = 0

count = 0
for i in EyeY.shape:
  count += 1
  if count == 2:
    x = int(i)
    
# loop through the trials

for index in range(0, x):
    
  q = p+index
  a =[]  
  for i in range(0,len(TargetY)):
    a.append(TargetX[i, index])
    if i > 1 and a[i] != a[i-1]:
      break
  target_jump = i
  del a, i 
  
  eyex = np.array(EyeX[:,index])
  tim = np.array(time_ms)
  eyey = np.array(EyeY[:,index])
  
  A = (np.diff(eyex) / np.diff(tim)) ** 2 + (np.diff(eyey) / np.diff(tim)) ** 2   
  Eye_velocity = sqrt(A)*1000
  Eye_velocity = np.append(Eye_velocity, 0)


  start = target_jump
  finish = len(EyeX)
  sooka = Eye_velocity[start:finish]

  init = TargetX[0, index]
  sac = TargetX[target_jump, index]

  if init <0:
    init = init*(-1)
    
  if sac<0:
    sac = sac*(-1);


  target_ampl = sac-init

  
  (saccade_onset, saccade_offset, peak_vel, saccade_duration, saccade_status) = saccade(Eye_velocity, vel_thres, down_vel_thres, sooka, start, new_max_sac_size)
  

  if saccade_status == 0:
    print('something wrong')
    
    Saccade['block_num'].append(block_num)
    Saccade['date'].append(which_date)
    Saccade['lum'].append(lum)
    Saccade['onset'].append(0)
    Saccade['offset'].append(0)
    Saccade['peak_velocity'].append(0)
    Saccade['duration'].append(0)
    Saccade['size'].append(0)
    Saccade['latency'].append(0)
    Saccade['init_targetX'].append(TargetX[0, index])
    Saccade['sac_targetX'].append(TargetX[target_jump, index])
    Saccade['target_jump'].append(target_jump)
    Saccade['target_ampl'].append(target_ampl)
    Saccade['dark_adapt'].append(da)
    Saccade['light_adapt'].append(la)
    Saccade['tgt_size'].append(tgt_size)
    
  else:
  
    x_off = EyeX[saccade_offset, index]
    x_on = EyeX[saccade_onset, index]  
    y_off = EyeY[saccade_offset, index]
    y_on = EyeY[saccade_onset, index]   
    saccade_size = sqrt((x_off - x_on)**2 + (y_off-y_on)**2)
  
    if saccade_size<0:
      saccade_size = saccade_size*(-1);

    Saccade['block_num'].append(block_num)
    Saccade['date'].append(which_date)
    Saccade['lum'].append(lum)
    Saccade['onset'].append(saccade_onset)
    Saccade['offset'].append(saccade_offset)
    Saccade['peak_velocity'].append(peak_vel)
    Saccade['duration'].append(saccade_duration)
    Saccade['size'].append(saccade_size)
    Saccade['latency'].append(saccade_onset-target_jump)
    Saccade['init_targetX'].append(TargetX[0, index])
    Saccade['sac_targetX'].append(TargetX[target_jump, index])
    Saccade['target_jump'].append(target_jump)
    Saccade['target_ampl'].append(target_ampl)
    Saccade['dark_adapt'].append(da)
    Saccade['light_adapt'].append(la)
    Saccade['tgt_size'].append(tgt_size)


  del saccade_duration, saccade_offset, saccade_onset, peak_vel, target_jump, sooka, start, saccade_size, saccade_status, target_ampl
  
  p += index

# plotting (new loop)
count = 0
for i in EyeY.shape:
  count += 1
  if count == 2:
    trial_num = int(i)
  
    

b = []
for index in range(0,trial_num):
  if b == 3:
    print('Кусенька моя любимая!')
    print('Я тебя очень люблю')
    break
  else:     
    if len(Saccade['onset']) == trial_num:
      q = index  
      
    elif len(Saccade['onset']) > trial_num:
      q = p+index   

    eyex = np.array(EyeX[:,index])
    tim = np.array(time_ms)
    eyey = np.array(EyeY[:,index])
  
    A = (np.diff(eyex) / np.diff(tim)) ** 2 + (np.diff(eyey) / np.diff(tim)) ** 2   
    Eye_velocity = sqrt(A)*1000
    Eye_velocity = np.append(Eye_velocity, 0)    
     

  ## plotting
  
    plotting1(EyeX, EyeY, Saccade, Eye_velocity, index, q, trial_num)
    
    while 1:
      b = int(input('next (1) correct (0) no saccade (2) or quit (3) '))
    
      if b == 2:
        
        if q == index:
          Saccade['onset'][q] = 0
          Saccade['offset'][q] = 0
          Saccade['peak_velocity'][q] = 0
          Saccade['duration'][q] = 0
          Saccade['size'][q] = 0
          Saccade['latency'][q] = 0

        else:
          Saccade['onset'][q-trial_num] = 0
          Saccade['offset'][q-trial_num] = 0
          Saccade['peak_velocity'][q-trial_num] = 0
          Saccade['duration'][q-trial_num] = 0
          Saccade['size'][q-trial_num] = 0
          Saccade['latency'][q-trial_num] = 0

     
      elif b == 0:
        print ('allright!')       
        print('select time interval in which the saccade will be detected')
        start = int(input ('input the start of the interval in ms '))
        finish = int(input ('input the end of the interval in ms '))


        while 1:
          if finish<=start:
            print('mistake!')         
            finish = int(input ('input the correct end of the interval in ms '))  
          else:
            sooka = Eye_velocity[start:finish]
            break

        new_max_sac_size = finish-start
   
        (saccade_onset, saccade_offset, peak_vel, saccade_duration, saccade_status) = saccade(Eye_velocity, vel_thres, down_vel_thres, sooka, start, new_max_sac_size)

         
        while 1:
          if saccade_onset == 0:
            print('error')
    
            while 1:
              c = int(input('change vel thres? (1 - change, 0 - no change) '))
              if c == 1:
                vel_thres = int(input ('what is the vel thres? '))
                down_vel_thres = int(input('what is the down vel thres? '))
                break
              elif c==0:
                break 

            start = int(input ('input the start of the interval in ms '))
            finish = int(input ('input the end of the interval in ms '))

    

            while 1:
              if finish<=start:
                print ('mistake!')
                finish = int(input ('input the correct end of the interval in ms '))
              else:
                sooka = Eye_velocity[start:finish]
                break

       
            new_max_sac_size = finish-start
     
            (saccade_onset, saccade_offset, peak_vel, saccade_duration, saccade_status) = saccade(Eye_velocity, vel_thres, down_vel_thres, sooka, start, new_max_sac_size)

          else:
            break 


        x_off = EyeX[saccade_offset, index]
        x_on = EyeX[saccade_onset, index]  
        y_off = EyeY[saccade_offset, index]
        y_on = EyeY[saccade_onset, index]   
        saccade_size = sqrt((x_off - x_on)**2 + (y_off-y_on)**2)

        if saccade_size<0:
          saccade_size = saccade_size*(-1)


        print('saccade onset is ' + str(saccade_onset) + ' ms')
        print('saccade offset is ' + str(saccade_offset) + ' ms')
        print('peak velocity is ' + str(peak_vel) + ' deg/sec')
        print('saccade duration is ' + str(saccade_duration) + ' ms')
        print('saccade size is ' + str(saccade_size) + ' deg')
  
        plotting2(EyeX, EyeY, Saccade, Eye_velocity, index, q, trial_num, saccade_onset, saccade_offset)

      # plt.show()
    
        if q == index:
          Saccade['onset'][q] = saccade_onset
          Saccade['offset'][q] = saccade_offset
          Saccade['peak_velocity'][q] = peak_vel
          Saccade['duration'][q] = saccade_duration
          Saccade['size'][q] = saccade_size
          Saccade['latency'][q] = saccade_onset- Saccade['target_jump'][q]
        else:
          Saccade['onset'][q-trial_num] = saccade_onset
          Saccade['offset'][q-trial_num] = saccade_offset
          Saccade['peak_velocity'][q-trial_num] = peak_vel
          Saccade['duration'][q-trial_num] = saccade_duration
          Saccade['size'][q-trial_num] = saccade_size
          Saccade['latency'][q-trial_num] = saccade_onset- Saccade['target_jump'][q-trial_num]

      elif b == 1 or b == 3:
        break
        
    
     

