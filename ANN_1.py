# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 17:38:57 2017

@author: Elias Wolf
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#importiere dataset

df = pd.read_excel('...')


print(df.info())
corr = df.iloc[:250,:].corr()

corr = abs(corr)
target_corr = corr.iloc[3:,1].sort_values(kind='mergesort', 
                       ascending= False)
print(target_corr)
rel = target_corr[0:20].index.values

print(rel.shape)

rel_features = rel.tolist()
print(rel_features)
    
    
# Graphics
graphs=['DEUTSCHE BANK (closing Price)', 'RVI', 'RSI', 'MFI', 
        'Hurst Exponent' ]

for i in graphs:
    plt.plot(df[i])
    plt.ylabel(i)
    plt.title('Time Series: '+i)
    plt.grid(True)
    plt.show()
    
    
# Data normalization

target = df.loc[:250,'Target']

predictors = df.loc[:250,rel_features]

print(predictors.info())



X=predictors.values
X_scaled = preprocessing.scale(X)
y=target.values
y=y.ravel()


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.1, random_state=42)


#Initial hyperparameters

penalty=0.0001
split=0.1
stop=0.0005

#Define ANN

clf = MLPClassifier(activation='relu', alpha=penalty, 
                    batch_size='auto', beta_1=0.9, beta_2=0.999,
                    early_stopping=False, epsilon=1e-08, 
                    hidden_layer_sizes=(50), learning_rate='constant',
                    learning_rate_init=0.001, max_iter=300, 
                    momentum=0.9, nesterovs_momentum=True,
                    power_t=0.5, random_state=1,
                    shuffle=True, solver='adam', tol=stop, 
                    validation_fraction=split, verbose=False,
                    warm_start=False)


#Optimize and train neural net

#Grid search parameter
tuned_parameters = [{'activation': ['relu'], 'alpha': [1e-2, 1e-4], 
                      'hidden_layer_sizes': [1,2,3,4,5,5,6,7,9,10]},
                      {'activation': ['relu'], 'alpha':[5e-2, 5e-4],
                      'hidden_layer_sizes': [1,2,3,4,5,5,6,7,9,10]}]


scores = ['accuracy', 'recall', 'precision' ]

best_models ={}

#Results 

for i in scores:
    clf_opt = GridSearchCV(clf, tuned_parameters, cv=5, 
                           scoring = i)
    clf_opt.fit(X_train,y_train)
    print('best model calibration based on: ' + i)
    print(clf_opt.best_params_)
    
    af = clf_opt.best_params_['activation']
    pen = clf_opt.best_params_['alpha']
    hls = clf_opt.best_params_['hidden_layer_sizes']
    

    mlp = MLPClassifier(activation= af, alpha= pen, 
                        batch_size='auto',
                        beta_1=0.9, beta_2=0.999, 
                        early_stopping=False,
                        epsilon=1e-08, hidden_layer_sizes= hls, 
                        learning_rate='constant',
                        learning_rate_init=0.001, max_iter=200, 
                        momentum=0.9, nesterovs_momentum=True,
                        power_t=0.5, random_state=1,
                        shuffle=True, solver='adam', tol=stop,
                        validation_fraction=split, 
                        verbose=False,warm_start=False)
                         
       

    mlp.fit(X_train,y_train)
    
    best_models[i.format(i)]=mlp

    plt.plot(mlp.loss_curve_)
    plt.title('Learning curve, selection criteria: ' + i)
    plt.ylabel('Loss function')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.show()
    
    quality = cross_val_score(mlp, X_train, y_train, scoring = i ,
                              cv=5)
    print(quality)
    print('Mean '+ i + ': ' +str(np.mean(quality)))
    
  
    
y_pred_1 = best_models['accuracy'].predict(X_test)
y_pred_2 = best_models['recall'].predict(X_test)
y_pred_3 = best_models['precision'].predict(X_test)


class_names = ['Down','Up']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), 
                                  range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix

cnf_matrix_1 = confusion_matrix(y_test, y_pred_1)
cnf_matrix_2 = confusion_matrix(y_test, y_pred_2)
cnf_matrix_3 = confusion_matrix(y_test, y_pred_3)
np.set_printoptions(precision=2)
    

print('accuracy:')

plt.figure()
plot_confusion_matrix(cnf_matrix_1, classes=class_names, 
                      normalize=True,
                      title='Normalized confusion matrix')
plt.show()
print(classification_report(y_test, y_pred_1, 
                            target_names=class_names))
print('accuracy: '+str(accuracy_score(y_test, y_pred_1)))


print('recall:')
plt.figure()
plot_confusion_matrix(cnf_matrix_2, classes=class_names, 
                      normalize=True,
                      title='Normalized confusion matrix')
plt.show()
print(classification_report(y_test, y_pred_2, 
                            target_names=class_names))
print('accuracy :'+str(accuracy_score(y_test, y_pred_2)))

print('precision:')
plot_confusion_matrix(cnf_matrix_3, classes=class_names, 
                      normalize=True,
                      title='Normalized confusion matrix')
plt.show()

print(classification_report(y_test, y_pred_3, 
                            target_names=class_names))
print('accuracy: '+str(accuracy_score(y_test, y_pred_3)))


