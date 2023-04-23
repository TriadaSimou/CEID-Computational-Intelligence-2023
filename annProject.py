import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import History 
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras import backend as K
from keras import regularizers
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

path = r"C:\Users\Trinity\Documents\ANN_project\dataset-HAR-PUC-Rio.csv" 
dataset= pd.read_csv(path, delimiter=";", decimal = ",", low_memory=False) # Read the file
#dataset.head()


oe = OrdinalEncoder()
oe.fit(dataset[["user","gender"]])
dataset[["user","gender"]] = oe.transform(dataset[["user","gender"]]) # Ordinal encoding of categorical input data

le = LabelEncoder()
dataset.Class = le.fit_transform(dataset.Class)# Label (integer) encoding of categorical target data
#dataset.head()

# Split the data to training and testing data 5-Fold
X = dataset.drop(["Class"], axis =1) # Input values
Y = dataset["Class"] # Target values


X = X.apply(lambda x: x-x.mean()) # Mean centering

scaler = MinMaxScaler()
X = scaler.fit_transform(X) # Scaling the data to [0,1]

kfold = StratifiedKFold(n_splits=5, shuffle =True) # Each fold has the same percentage of samples for every class

# Initializing lists
crossentropyList = []
accuracyList = []
mseList = []

accuracyhistoryList = []
val_acchistoryList = []
losshistoryList= []
val_losshistoryList = []
val_msehistoryList = []


# KFold loop
for i, (train, test) in enumerate(kfold.split(X,Y)):
    
    # Create model
    model = Sequential() 
    
    model.add(Dense(23, activation="relu", kernel_regularizer = regularizers.l2(0.9), input_dim=18)) #  Dense = fully connected
    model.add(Dense(5, activation="softmax", input_dim=23)) #  Softmax for crossentropy loss function

    # Compile model
    keras.optimizers.SGD(learning_rate=0.1, momentum=0.6) # Stochastic gradient descent optimizer
    model.compile(optimizer = 'sgd', loss = 'sparse_categorical_crossentropy', metrics=['accuracy','mean_squared_error']) # Sparse categorical crossentropy loss for integer encoding 

    # Callback for EarlyStopping
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=0, patience=10) # When accuracy maximizes it waits another 10 epochs, it stops if there is no change. 
    #If there is, it continues until the next plateau or until it reaches the epochs intilized.
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)# Saves the model with the best performance obserbed during training
    history = History()
    
    # Fit model
    model.fit(X[train], Y[train], validation_data =(X[test], Y[test]), batch_size= 10, epochs=30, verbose=1, callbacks= [es, mc, history]) # Validation with test dataset 
    
    # Load best model
    saved_model = load_model('best_model.h5')
    
    # Append trainig histories to lists
    accuracyhistoryList.append(history.history['accuracy'])
    val_acchistoryList.append(history.history['val_accuracy'])
    losshistoryList.append(history.history['loss'])
    val_losshistoryList.append(history.history['val_loss'])
    val_msehistoryList.append(history.history['val_mean_squared_error'])
    
    # Evaluate model
    scores = saved_model.evaluate(X[test], Y[test], verbose=0) # The "best model" is evaluated
    crossentropyList.append(scores[0])
    accuracyList.append(scores[1])
    mseList.append(scores[2])
    print("Fold :", i, " Test Loss:", scores[0], " Test Accuracy:", scores[1], " Test MSE:", scores[2])

# History of "average model"
avg_acc_hist =[]
avg_val_acc_hist=[]
avg_loss_hist =[]
avg_val_loss_hist =[]
avg_val_mse_history=[]
avg_acc_hist=np.mean(accuracyhistoryList, axis=0)
avg_val_acc_hist=np.mean(val_acchistoryList, axis=0)
avg_loss_hist=np.mean(losshistoryList, axis=0)
avg_val_loss_hist=np.mean(val_losshistoryList, axis=0)
avg_val_mse_hist=np.mean(val_msehistoryList, axis=0)

# Plot graphs for average model

# Summarize history for accuracy
plt.plot(avg_acc_hist)
plt.plot(avg_val_acc_hist)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(avg_loss_hist)
plt.plot(avg_val_loss_hist)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for mse
plt.plot(val_msehistoryList[i])
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['test'], loc='upper left')
plt.show()



 