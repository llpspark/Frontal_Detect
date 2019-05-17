#coding:utf-8

#> run like:
#>: 
# python2 classify.py 20 '/home/spark/grocery/FER/codes/caffe_2classify/test/test_img.txt' '/home/spark/grocery/FER/codes/caffe_2classify/test/res.txt'

import numpy as np
import sys, os
sys.path.append('/home/spark/caffe-master/python')
import caffe
from multiprocessing import Process


def single_classify(proc_num, proc_id, imgs_text, save_txt):
  net_file = '/home/spark/grocery/FER/codes/caffe_2classify/test/size32_2classify_deploy.prototxt'
  caffe_model = "/home/spark/grocery/FER/codes/caffe_2classify/model/size32_2classify_v3_iter_49500.caffemodel"
  net = caffe.Net(net_file, caffe_model, caffe.TEST)
  caffe.set_mode_gpu()

  with open(imgs_text, 'r') as fr:
    for i, img_and_label in enumerate(fr.readlines()):
      if not i % proc_num == proc_id:
        continue
      img_path = img_and_label.strip('\n').split(' ')[0]
      transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
      transformer.set_transpose('data', (2, 0, 1))
      transformer.set_raw_scale('data', 255)
      #transformer.set_channel_swap('data', (2, 1, 0))
      net.blobs['data'].reshape(1, 1, 32, 32)
      
      #im = caffe.io.load_image(img_path) 
      im = caffe.io.load_image(img_path, False) 
      net.blobs['data'].data[...] = transformer.preprocess('data', im)
      output = net.forward()
       
      output_prob = output['out'][0]
      with open(save_txt, 'a') as fw:
        fw.write(img_and_label.strip('\n') +  ' ' + str(output_prob.argmax()) + " " + \
        str(output_prob[0]) + ' ' + str( output_prob[1]) + '\n')


class Multi_Classify(Process):
  def __init__(self, proc_num, proc_id, imgs_text, save_txt):
    super(Multi_Classify, self).__init__()
    self.proc_num = proc_num
    self.proc_id = proc_id
    self.imgs_text = imgs_text
    self.save_txt = save_txt

  def run(self):
    single_classify(self.proc_num, self.proc_id, self.imgs_text, self.save_txt)


def batch_classify(proc_num, imgs_text, save_txt):
  proc_list = []
  for i in range(proc_num):
    proc_list.append(Multi_Classify(proc_num, i, imgs_text, save_txt))

  for proc in proc_list:
    proc.start()

  for proc in proc_list:
    proc.join()

if __name__ == '__main__':
  proc_num = int(sys.argv[1])
  imgs_text = sys.argv[2]
  save_txt = sys.argv[3]
  batch_classify(proc_num, imgs_text, save_txt)
  print("ok")
