import pandas as pd
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from sklearn import preprocessing
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import itertools
import numpy as np



def main():
    

    predict_model()
   





def predict_model(airport=None):


    file = open("model_result-" + str(airport) + ".txt","w+")

    # read the processed dataset

    flights_data = pd.read_csv('2018-flData_2.csv')

    
    # if airport name is not given then calculate prediction for all airports
    if airport != None:

        flights_data = flights_data[(flights_data['DEST'] == airport)]
    

    
    print("--------------------------------------------------------------- \n\n" + str(flights_data.shape)
    + "\n\n--------------------------------------------------------------- "
     )
    

    status = []

  
    # drop the columns that are not being used
    flights_data = flights_data.drop(['ARR_DELAY','FL_DATE','OP_CARRIER','ORIGIN','DEST',
    'CRS_ARR_TIME','CRS_DEP_TIME','DAY','MONTH','YEAR'], axis=1)
    
    df = flights_data.astype(float)

    #drop_mon = ['MONTH_1','MONTH_2','MONTH_3','MONTH_4','MONTH_4','MONTH_5','MONTH_6','MONTH_7','MONTH_8','MONTH_9','MONTH_10','MONTH_11','MONTH_12']

    #flights_data.info()

    col_names = list(df.columns)

    # drop the un-named columns to remove any noise in the data
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    y = df[['DELAY_BIN','MONTH_1','MONTH_2','MONTH_3','MONTH_4','MONTH_4','MONTH_5',
    'MONTH_6','MONTH_7','MONTH_8','MONTH_9','MONTH_10','MONTH_11','MONTH_12']]
    
    X = df.drop(['DELAY_BIN','OP_CARRIER_FL_NUM'], axis=1)

    col_sc = list(df.columns)

    s_scaler = preprocessing.StandardScaler()
    
    scaled_data = s_scaler.fit_transform(df)

    scaled_data= pd.DataFrame(scaled_data, columns=col_sc)
    


    # Split the data set, in here training is done on January (1) to November (11), testing is done on December
    X_train = X[X['MONTH_12'] != 1]
    y_train = y[X['MONTH_12'] != 1]
    y_train = y_train['DELAY_BIN']

    X_test = X[X['MONTH_12'] == 1]
    y_test = y[X['MONTH_12'] == 1]
    y_test = y_test['DELAY_BIN']

    df.info()
    lstm_out = 200
    nn_mod = Sequential()
    n_features = X_train.shape[1]
    print(str(n_features))

    #nn_mod.add(Dense(8, activation='relu', input_shape=(77,)))
    #nn_mod.add(Embedding(40,input_length=77,output_dim=2))


    #nn_mod.add(Dense(4,  activation='tanh'))

    # Adding the layers to neural network, MLP Binary classification

    nn_mod.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
   
    nn_mod.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
    
    nn_mod.add(Dense(4, activation='tanh', kernel_initializer='he_normal'))
    # since it's binary, sigmoid is used at the end
    nn_mod.add(Dense(1, activation='sigmoid'))

    #nn_mod.add(Dense(1, activation='sigmoid'))

    nn_mod.summary()

    # compile to finalize the model here
    nn_mod.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fitting the model
    nn_mod.fit(X_train, y_train, epochs=20, batch_size=24, validation_split=0.3)

    # passing the test dataset to predict
    y_pred = nn_mod.predict(X_test)
    # since the output of sigmoid is probability, taking only probs which is greater than 0.5
    y_pred_1 =(y_pred > 0.5)


    # converting the predicted values to 0 and 1 from true and false, where 0 means no delay
    x_out = np.where(y_pred > 0.5, 1,0)

    # count the number of 0's and 1's
    unique, counts = np.unique(x_out, return_counts=True)
    # store it in dictionary
    my_dict=dict(zip(unique, counts))
    # calculate the percentage of flights that are predicted to be delayed in test dataset( December)
    delay_predicted =     ( my_dict.get(1) / ( my_dict.get(0) + my_dict.get(1) ) ) * 100

    print(str(my_dict))

    print(str(delay_predicted ))



    print(str(x_out))

    

    # calculating the accuracy
    model_accuracy = metrics.accuracy_score(y_test, y_pred_1)
    
   
    
    print('Accuracy:', round(model_accuracy*100),'%')
    
    file.write(str(round(model_accuracy*100)) + "\n\n")

    precision = precision_score(y_test, y_pred_1)

    file.write(str(round(precision*100, 2)))

    print('Precision score:', round(precision*100, 2),'%')

    #print(str(y_pred))
    # plotting the confusion matrix, the function was copied from the official docs of scikit
    cm = confusion_matrix(y_test,y_pred_1)
    
    cm_plot_labels = ['Delayed','Not Delayed']
    
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix',fileName=airport)

    return str(delay_predicted)

    #df_s.to_csv("df_s.csv")


   
# reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues,
                        fileName=None
                        ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if fileName != None:
        plt.savefig('conf_matrixAIRPORT-' + fileName + '.png')
    plt.show()
    plt.close()
    

    






if __name__ == "__main__":
    main()
        