# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 07:30:32 2025

@author: User
"""

# import libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# baca dataset
df = pd.read_csv('data_kartu_kredit.csv')
df.shape


# melakukan standarisasi kolom amount
from sklearn.preprocessing import StandardScaler
df['std_amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))

# menentukan variable dependent (y) dan variable independent (X)
y = np.array(df.iloc[:, -2])
X = np.array(df.drop(columns=['Time', 'Amount', 'Class'], axis=1))

# membagi training dan test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                    test_size=0.2, 
                                                    random_state=111)

X_train, X_validate, y_train, y_validate = train_test_split(X_train,y_train, 
                                                    test_size=0.2, 
                                                    random_state=111)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# dense = untuk menentukan jumlah neuron yang di tambahkan disetiap layer
# dropout = sub library untuk menentukan probabilitas hilangnya nodes secara random

classifier = Sequential() # model

'''
note : 
    - input_dim = jika 2 dimensi
    - input_shape = jika lebih dari itu
    
activation function :
    - input --> hidden = relu 
    - output layer = sigmoid
'''
classifier.add(Dense(units=16, input_dim=29, activation='relu')) # input layer
classifier.add(Dense(24, activation='relu')) # hidden layer

'''
- Dropout = teknik untuk menonaktifkan neuron/node
- untuk meminimalisir overfitting dengan menggunakan Dropout
- angka yang dimasukan persentasi probabilitas node/saraf-nya yang akan di matikan secara random
'''
classifier.add(Dropout(0.25))
classifier.add(Dense(20, activation='relu')) # hidden layer
classifier.add(Dense(24, activation='relu')) # hidden layer

'''
output layer, parameter unitsnya wajib bernilai 1
'''
classifier.add(Dense(1, activation='sigmoid')) # output layer

'''
- parameter optimizer (Stochastic Gradient Descent) = merupakan algoritma untuk menentukan posisi lembah terendah
- biasanya yang di pakai nilainya itu 'adam'
- loss (function) = binary_crossentropy menjadi pertimbangan utama, setiap training/setiap 1 epoch dihitung loss functionnya untuk menentukan apakah model yang dibuat sudah cukup baik atau belum
- metrics = proses evaluasi suatu model tetapi tidak menjadi tolak ukur utama untuk menentukan baik tidaknya model yang dibuat
'''
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.summary()

# visualisasi model ann
from tensorflow.keras.utils import plot_model
plot_model(classifier, to_file='model_ann.png', show_shapes=True, show_layer_names=False)


# proses training model ann kita
run_model = classifier.fit(X_train,y_train, batch_size=32,
                           epochs=5, verbose=1,
                           validation_data=(X_validate, y_validate))

# melihat parameter apa saja yang disimpan
print(run_model.history.keys())

# plot accuracy dengan val_accuracy
plt.plot(run_model.history['accuracy'])
plt.plot(run_model.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('accurcay')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

# plot loss dengan val_loss
plt.plot(run_model.history['loss'])
plt.plot(run_model.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()


# mengevaluasi model ann
evaluation = classifier.evaluate(X_test, y_test)
print('accuracy : {:.2f}'.format(evaluation[1]*100))

# membuat prediksi test set
pred_result= (classifier.predict(X_test) > 0.5).astype("int32")
classes_X= np.argmax(pred_result, axis=1)


# membuat confusion matrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred_result)
cm_label=pd.DataFrame(cm,columns=np.unique(y_test), index=np.unique(y_test))
cm_label.index.name = 'Actual'
cm_label.columns.name ='Prediction'

sns.heatmap(cm_label, annot=True, cmap='Reds', fmt='g')

# membuat classification_repot
from sklearn.metrics import classification_report
category_amount = 2
target_names = ['Class {}'.format(i) for i in range(category_amount)]
print(classification_report(y_test, pred_result, target_names = target_names))


# melakukan under sampling
idx_fraud = np.array(df[df.Class==1].index)
n_fraud = len(idx_fraud)

idx_normal = np.array(df[df.Class==0].index)
idx_data_normal = np.random.choice(idx_normal, n_fraud, replace=False)
idx_merge_data=np.concatenate([idx_fraud, idx_data_normal])

new_df=df.iloc[idx_merge_data,:]

# menentukan variable dependent (y) dan variable independent (X)
new_y = np.array(new_df.iloc[:, -2])
new_X = np.array(new_df.drop(columns=['Time', 'Amount', 'Class'], axis=1))


# membagi train dan test set
X_train2, X_test_final, y_train2, y_test_final = train_test_split(new_X, new_y, test_size=0.1, random_state=111)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train2, y_train2, test_size=0.2, random_state=111)
X_train2, X_validate2, y_train2, y_validate2 = train_test_split(X_train2, y_train2, test_size=0.2, random_state=111)


# merancang ann yang baru (model balanced)
classifier2 = Sequential() # model

'''
note : 
    - input_dim = jika 2 dimensi
    - input_shape = jika lebih dari itu
    
activation function :
    - input --> hidden = relu 
    - output layer = sigmoid
'''
classifier2.add(Dense(units=16, input_dim=29, activation='relu')) # input layer
classifier2.add(Dense(20, activation='relu')) # hidden layer

'''
- Dropout = teknik untuk menonaktifkan neuron/node
- untuk meminimalisir overfitting dengan menggunakan Dropout
- angka yang dimasukan persentasi probabilitas node/saraf-nya yang akan di matikan secara random
'''
classifier2.add(Dropout(0.25))
classifier2.add(Dense(20, activation='relu')) # hidden layer
classifier2.add(Dense(24, activation='relu')) # hidden layer

'''
output layer, parameter unitsnya wajib bernilai 1
'''
classifier2.add(Dense(1, activation='sigmoid')) # output layer

'''
- parameter optimizer (Stochastic Gradient Descent) = merupakan algoritma untuk menentukan posisi lembah terendah
- biasanya yang di pakai nilainya itu 'adam'
- loss (function) = binary_crossentropy menjadi pertimbangan utama, setiap training/setiap 1 epoch dihitung loss functionnya untuk menentukan apakah model yang dibuat sudah cukup baik atau belum
- metrics = proses evaluasi suatu model tetapi tidak menjadi tolak ukur utama untuk menentukan baik tidaknya model yang dibuat
'''
classifier2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier2.summary()

# proses training model ann baru (model balanced data)
run_model2 = classifier.fit(X_train2,y_train2, batch_size=32,
                           epochs=10, verbose=1,
                           validation_data=(X_validate2, y_validate2))


# melihat parameter apa saja yang disimpan
print(run_model2.history.keys())

# plot accuracy dengan val_accuracy
plt.plot(run_model2.history['accuracy'])
plt.plot(run_model2.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('accurcay')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

# plot loss dengan val_loss
plt.plot(run_model2.history['loss'])
plt.plot(run_model2.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

# mengevaluasi model ann
evaluation2 = classifier2.evaluate(X_test2, y_test2)
print('accuracy2 : {:.2f}'.format(evaluation2[1]*100))

# membuat prediksi test set
pred_result2= (classifier2.predict(X_test2) > 0.5).astype("int32")
classes_X2= np.argmax(pred_result2, axis=1)


# membuat confusion matrics
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test2, pred_result2)
cm_label2=pd.DataFrame(cm2,columns=np.unique(y_test2), index=np.unique(y_test2))
cm_label2.index.name = 'Actual'
cm_label2.columns.name ='Prediction'

sns.heatmap(cm_label2, annot=True, cmap='Reds', fmt='g')

# membuat classification_repot
from sklearn.metrics import classification_report
category_amount2 = 2
target_names2 = ['Class {}'.format(i) for i in range(category_amount2)]
print(classification_report(y_test2, pred_result2, target_names = target_names2))

# membuat prediksi test set final
pred_result3= (classifier2.predict(X_test_final) > 0.5).astype("int32")
cm2 = confusion_matrix(y_test_final, pred_result3)
cm_label3=pd.DataFrame(cm2,columns=np.unique(y_test_final), index=np.unique(y_test_final))
cm_label3.index.name = 'Actual'
cm_label3.columns.name ='Prediction'
sns.heatmap(cm_label3, annot=True, cmap='Reds', fmt='g')


