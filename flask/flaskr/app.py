import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import os


from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

global loaded_model
global dataset
df = pd.read_csv(r'C:\Users\nitro\git\Deep-Learning-Hotel-Res\clean_data.csv')

def preprocessData(df):
    cleandf = df[['lead_time','avg_price_per_room','no_of_special_requests','arrival_date','arrival_month','market_segment_type','no_of_week_nights','booking_status']]
    categorical_cols = cleandf.select_dtypes(include='object').columns
    le = LabelEncoder()
    cleandf[categorical_cols] = cleandf[categorical_cols].apply(lambda col: le.fit_transform(col))
    one_hot_scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(cleandf), columns=cleandf.columns)
    dataset = one_hot_scaled_df
    return one_hot_scaled_df

def trainModel():
    df = pd.read_csv(r'C:\Users\nitro\git\Deep-Learning-Hotel-Res\Hotel Reservations.csv')
    df = preprocessData(df)
    dataset = df
    X = df.drop(['booking_status'], axis=1)
    y = df['booking_status']
    oversample=SMOTE()
    X, y= oversample.fit_resample(X,y)
    counter=Counter(y)
    # Split the data into training and testing sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1)
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(256, activation='relu', input_dim=X_train.shape[1]))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    path_checkpoint = r"C:\Users\nitro\git\Deep-Learning-Hotel-Res\flask\training\cp.ckpt"
    directory_checkpoint = os.path.dirname(path_checkpoint)

    callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,
                                                 save_weights_only=True,
                                                 verbose=1)

    history = model.fit(x = X_train, y = y_train, validation_data = (X_val, y_val),  batch_size = 256, epochs = 100, callbacks=[callback])

    model.save(r'C:\Users\nitro\git\Deep-Learning-Hotel-Res\flask\saved_models\model1')
    print('Model Created and Saved')
    loaded_model = tf.keras.models.load_model(r'C:\Users\nitro\git\Deep-Learning-Hotel-Res\flask\saved_models\model1')
    print('Model loaded')
    print(loaded_model.summary())

def modelInference(x,path = r'C:\Users\nitro\git\Deep-Learning-Hotel-Res\flask\saved_models\model1'):
    loaded_model = tf.keras.models.load_model(path)
    y =[0.11963883, 0.32833333, 0.4       , 0.66666667, 0.27272727,
       1.        , 0.17647059]
    prediction = loaded_model.predict(np.array( [x,] ) )
    print(prediction,prediction.round())

x = [0.11963883, 0.32833333, 0.4       , 0.66666667, 0.27272727,
       1.        , 0.17647059]
# trainModel()
modelInference(x)

