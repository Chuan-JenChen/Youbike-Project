#!/usr/bin/env python
# coding: utf-8

# In[4]:


import json
import numpy as np
import tensorflow as tf
import math


# In[251]:


with open(r'C:\Users\USER\Desktop\correct\patient_data6var4hrX2.json', 'r') as f:
    patient_data = json.load(f)




data_size = len(patient_data)
x_train = []
y_train = []

for ii in range(1, 151):
    data_iter = []                      
    data_FiO2 = patient_data['person_{}'.format(ii)]['FiO2']
    #print(6,len(data_FiO2))
    data_PEEP = patient_data['person_{}'.format(ii)]['PEEP']
    #print(5,len(data_PEEP))
    data_Pmean = patient_data['person_{}'.format(ii)]['Pmean']
    #print(4,len(data_Pmean))
    data_Ppeak = patient_data['person_{}'.format(ii)]['Ppeak']
    #print(3,len(data_Ppeak))
    data_RR = patient_data['person_{}'.format(ii)]['RR']
    #print(2,len(data_RR))
    data_Vte = patient_data['person_{}'.format(ii)]['Vte']
    #print(1,len(data_Vte))
    data_FiO2 = (data_FiO2 - np.mean(data_FiO2)) #/ np.linalg.norm(data_FiO2)
    data_PEEP = (data_PEEP - np.mean(data_PEEP)) #/ np.linalg.norm(data_PEEP)
    data_Pmean = (data_Pmean - np.mean(data_Pmean)) #/ np.linalg.norm(data_Pmean)
    data_Ppeak = (data_Ppeak - np.mean(data_Ppeak)) #/ np.linalg.norm(data_Ppeak)
    data_RR = (data_RR - np.mean(data_RR))  #/np.linalg.norm(data_RR)
    data_Vte = (data_Vte - np.mean(data_Vte)) #/ np.linalg.norm(data_Vte)

    data_iter.append(data_FiO2)
    data_iter.append(data_PEEP)
    data_iter.append(data_Pmean)
    data_iter.append(data_Ppeak)
    data_iter.append(data_RR)
    data_iter.append(data_Vte)
    data_iter = np.array(data_iter, dtype=np.float32)
    
    index = np.isnan(data_iter)
    data_iter[index] = 0
    data_label = np.array(patient_data['person_{}'.format(ii)]['Succ'], dtype=np.float32)
    x_train.append(data_iter)
    y_train.append(data_label)   

x_train = np.array(x_train)
y_train = np.array(y_train)


# In[3]:


# #for ii in range(1, 151):
# #    data_iter = []
#  #   data_FiO2 = patient_data['person_{}'.format(ii)]['FiO2']
#   #  #print(6,len(data_FiO2))
#    # data_PEEP = patient_data['person_{}'.format(ii)]['PEEP']
#     #print(5,len(data_PEEP))
#     data_Pmean = patient_data['person_{}'.format(ii)]['Pmean']
#     #print(4,len(data_Pmean))
#     data_Ppeak = patient_data['person_{}'.format(ii)]['Ppeak']
#     #print(3,len(data_Ppeak))
#     data_RR = patient_data['person_{}'.format(ii)]['RR']
#     #print(2,len(data_RR))
#     data_Vte = patient_data['person_{}'.format(ii)]['Vte']
#     #print(1,len(data_Vte))
#     data_FiO2 = (data_FiO2 - np.mean(data_FiO2)) #/ np.linalg.norm(data_FiO2)
#     data_PEEP = (data_PEEP - np.mean(data_PEEP)) #/ np.linalg.norm(data_PEEP)
#     data_Pmean = (data_Pmean - np.mean(data_Pmean)) #/ np.linalg.norm(data_Pmean)
#     data_Ppeak = (data_Ppeak - np.mean(data_Ppeak)) #/ np.linalg.norm(data_Ppeak)
#     data_RR = (data_RR - np.mean(data_RR)) #/ np.linalg.norm(data_RR)
#     data_Vte = (data_Vte - np.mean(data_Vte)) #/ np.linalg.norm(data_Vte)
    
#     data_iter.append(data_FiO2)
#     data_iter.append(data_PEEP)
#     data_iter.append(data_Pmean)
#     data_iter.append(data_Ppeak)
#     data_iter.append(data_RR)
#     data_iter.append(data_Vte)
#     data_iter = np.array(data_iter, dtype=np.float32)
    
#     index = np.isnan(data_iter)
#     data_iter[index] = 0
#     data_label = np.array(patient_data['person_{}'.format(ii)]['Succ'], dtype=np.float32)
#     x_train.append(data_iter)
#     y_train.append(data_label)
    
# x_train = np.array(x_train)
# y_train = np.array(y_train)


# In[7]:


x_train.shape


# In[8]:



x_test = []
y_test = []
for ii in range(151,201):
    data_iter = []
    data_FiO2 = patient_data['person_{}'.format(ii)]['FiO2']
    #print(6,len(data_FiO2))
    data_PEEP = patient_data['person_{}'.format(ii)]['PEEP']
    #print(5,len(data_PEEP))
    data_Pmean = patient_data['person_{}'.format(ii)]['Pmean']
    #print(4,len(data_Pmean))
    data_Ppeak = patient_data['person_{}'.format(ii)]['Ppeak']
    #print(3,len(data_Ppeak))
    data_RR = patient_data['person_{}'.format(ii)]['RR']
    #print(2,len(data_RR))
    data_Vte = patient_data['person_{}'.format(ii)]['Vte']
    #print(1,len(data_Vte))
    data_FiO2 = (data_FiO2 - np.mean(data_FiO2)) #/ np.linalg.norm(data_FiO2)
    data_PEEP = (data_PEEP - np.mean(data_PEEP)) #/ np.linalg.norm(data_PEEP)
    data_Pmean = (data_Pmean - np.mean(data_Pmean)) #/ np.linalg.norm(data_Pmean)
    data_Ppeak = (data_Ppeak - np.mean(data_Ppeak)) #/ np.linalg.norm(data_Ppeak)
    data_RR = (data_RR - np.mean(data_RR))  #/np.linalg.norm(data_RR)
    data_Vte = (data_Vte - np.mean(data_Vte)) #/ np.linalg.norm(data_Vte)
 
    data_iter.append(data_FiO2)
    data_iter.append(data_PEEP)
    data_iter.append(data_Pmean)
    data_iter.append(data_Ppeak)
    data_iter.append(data_RR)
    data_iter.append(data_Vte)
    data_iter = np.array(data_iter,dtype=np.float32)
    
    index = np.isnan(data_iter)
    data_iter[index] = 0
    data_label = np.array(patient_data['person_{}'.format(ii)]['Succ'], dtype=np.float32)
    x_test.append(data_iter)
    y_test.append(data_label)

x_test = np.array(x_test)
y_test = np.array(y_test)


# In[9]:


x_test.shape


# In[10]:


# #y_trans=[]
# for i in range (0,len(y_train)):
#     if y_train[i] == 0 :
#         y = [1,0]
#         y_trans.append(y)
#     if y_train[i]== 1 :
#         y = [0,1]
#         y_trans.append(y)

# y_trans=np.array(y_trans)


# In[244]:


#CNN 設定層數
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),

  tf.keras.layers.Dense(128, activation='sigmoid'),

  tf.keras.layers.Dense(32, activation='sigmoid'),


  tf.keras.layers.Dense(4, activation='sigmoid'),
  tf.keras.layers.Dense(2, activation='softmax')])


#CNN 選擇優化器，learning_rate
opt = tf.keras.optimizers.Adam(learning_rate=0.005)
print(x_train.shape)
print(y_trans.shape)
model.compile(optimizer=opt,
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

model.fit(x_train, y_trans, epochs=1000)
print(model(x_train).numpy())
print(y_trans)

 
#LSTM  設定層數
model = tf.keras.models.Sequential([

  tf.keras.layers.LSTM(30, return_sequences=True),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.LSTM(20, return_sequences=True),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.LSTM(10),
  tf.keras.layers.Dense(2, activation='softmax')])


#LSTM  選擇優化器，learning_rate
opt = tf.keras.optimizers.Adam(learning_rate=0.005)
print(x_train.shape)
print(y_trans.shape)
model.compile(optimizer=opt,
                loss='mse', 
                metrics=['accuracy'])

model.fit(x_train, y_trans, epochs=500)
print(model(x_train).numpy())
print(y_trans)


# 要再重新寫
j = 0
for ii in range (0,50):
    predictions = model(x_test[ii:ii+1]).numpy()
    ans = tf.nn.softmax(predictions).numpy()
    if abs(np.argmax(ans) - y_test[ii]) < 0.5:
        j = j + 1
    print(np.argmax(ans), y_test[ii])
print('測試資料集：共50筆')
print('成功率：' + str(j/50*100) + '%')






