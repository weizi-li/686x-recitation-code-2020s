### enforce to use CPU only
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

### import auxiliary libraries
import numpy
import math
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

### import keras and relevant functions
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K



#########################
#########################
### LSTM example
#########################
#########################
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


def prepare_dataset(look_back=1):
    # load the dataset
    dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # reshape into X=t and Y=t+1
    x_train, y_train = create_dataset(train, look_back)
    x_test, y_test = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    x_train = numpy.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = numpy.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    return x_train, y_train, x_test, y_test, dataset, scaler

def run_lstm():
    # fix random seed for reproducibility
    numpy.random.seed(42)
    x_train, y_train, x_test, y_test, dataset, scaler = prepare_dataset()

    # create and fit the LSTM network
    ### TO be implemented

    # make predictions
    p_train = model.predict(x_train)
    p_test = model.predict(x_test)

    # invert predictions
    p_train = scaler.inverse_transform(p_train)
    y_train = scaler.inverse_transform([y_train])
    p_test = scaler.inverse_transform(p_test)
    y_test = scaler.inverse_transform([y_test])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train[0], p_train[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[0], p_test[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))


#########################
#########################
### CNN example
#########################
#########################
def save_my_model(model):
    ### TO be implemented
    return 0


def new_model(x_train, y_train, x_test, y_test, input_shape, batch_size, num_classes, epochs):
    ### TO be implemented
    return 0


def existing_model(x_test, y_test):
    ### TO be implemented
    return 0


def run_cnn():
    ### load the data
    ### TO be implemented

    ### input image dimensions
    img_rows, img_cols = 28, 28

    ### specify the input_shape for later use
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    ### design a new model and train it
    #new_model(x_train, y_train, x_test, y_test, input_shape, batch_size, num_classes, epochs)

    ### use an existing model
    #existing_model(x_test, y_test)


if __name__ == "__main__":
    run_cnn()
    #run_lstm()








