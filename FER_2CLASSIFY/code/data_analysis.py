#> run like:
#>  python data_analysis.py /home/spark/grocery/FER/codes/caffe_2classify/test/res.txt 

import os,sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def modify_val(s):
  if s < 0.5:
    return (1 - s)
  else:
    return s


def stage_percent_eval(input_file, stage):
  pd.set_option('display.width', 500)
  try:
    df = pd.read_table(input_file, delimiter = ' ', header = None)
  except:
    print('read error!')
  
  plt.ion()
  ###-----------positive sample analysis--------###
  df_pos = df[df[1] == 0]
  #df_modval = df[3].apply(lambda x: modify_val(x))
  total_pos = df_pos.shape[0]
  print('total is', total_pos)
  pers = []
  porports = []
  for i in range(len(stage) - 1):
    per_df = df_pos[df_pos[3] <= stage[i+1]]
    porpo_num = per_df.shape[0]
    match_num = per_df.shape[0] - per_df[2].astype(int).sum()
    pers.append((match_num / porpo_num) * 100.0)
    porports.append((porpo_num / total_pos) * 100.0)

  fig1 = plt.figure()
  ax1 = fig1.add_subplot(111)
  ax1.plot(stage[1:], pers, label = 'match rate')
  ax1.plot(stage[1:], porports, label = 'confidence porportion')
  ax1.legend()

  ax1.set_xlabel('positive class confidence')
  ax1.set_ylabel('rate (%)')
  #plt.ioff()
  #plt.show()
  #plt.savefig('./percent_00.jpg')

  ###------------negative sample analysis--------------###
  df_neg = df[df[1] == 1]
  #df_modval = df[3].apply(lambda x: modify_val(x))
  total_neg = df_neg.shape[0]
  pers1 = []
  porports1 = []
  for i in range(len(stage) - 1):
    per_df1 = df_neg[df_neg[4] <= stage[i+1]]
    porpo_num1 = per_df1.shape[0]
    match_num1 = per_df1[2].astype(int).sum()
    pers1.append((match_num1 / porpo_num1) * 100.0)
    porports1.append((porpo_num1 / total_pos) * 100.0)

  fig2 = plt.figure()
  ax2 = fig2.add_subplot(111)
  ax2.plot(stage[1:], pers1, label = 'match rate')
  ax2.plot(stage[1:], porports1, label = 'confidence porportion')
  ax2.legend()

  ax2.set_xlabel('negative class confidence')
  ax2.set_ylabel('rate (%)')
  plt.ioff()
  plt.show()


def stage_match_rate(input_file, stage):
  pd.set_option('display.width', 500)
  try:
    df = pd.read_table(input_file, header = None, delimiter = ' ')
  except:
    print('read error!')
  
  total = df.shape[0]
  df[3] = df[3].apply(lambda x : modify_val(x))
  
  match_rate = []
  for i in range(len(stage) - 1):
    df_thresh1 = df[df[3] < stage[i+1]]
    df_thresh2 = df_thresh1[df_thresh1[3] > stage[i]]
    stage_total =  df_thresh2.shape[0]
    match_num = (df_thresh2[1] ^ df_thresh2[2]).astype(int).sum()
    match_rate.append(100 - 100 * match_num / stage_total)

  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax1.plot(stage[1:], match_rate)
  ax1.set_xlabel('classify confidence')
  ax1.set_ylabel('match(pass) rate (%)')
  plt.show()
    

def thres_filter(input_file, thres_0, thres_1):
  pd.set_option('display.width', 500)
  try:
    df = pd.read_table(input_file, header = None, delimiter = ' ')
  except:
    print('read error!')
  #---filter positive sample----#
  df_pos = df[df[1] == 0]
  pos_filt = df_pos[df_pos[3] > thres_0]
  pos_save = pd.DataFrame({'col0':pos_filt[0], 'col1':pos_filt[1]})
  pos_save.to_csv('./0.txt', header=None, index=False, sep=' ')

  #---fileter negative sample---#
  df_neg = df[df[1] == 1]
  neg_filt = df_neg[df_neg[4] > thres_1]
  neg_save = pd.DataFrame({'col0':neg_filt[0], 'col1':neg_filt[1]})
  neg_save.to_csv('./1.txt', header=None, index=False, sep=' ')

if __name__ == '__main__':
  input_file = sys.argv[1]
  stage = [i for i in np.arange(0.5, 1.01, 0.01)]
  #stage_percent_eval(input_file, stage)
  #stage_match_rate(input_file, stage)
  thres_filter(input_file, 0.98, 0.99)
