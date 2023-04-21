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
df = pd.read_csv(r'E:\George Brown College Semester 1 2\SEM 2\Full stack\Project\Deep-Learning-1\clean_data.csv')

def preprocessData(df):
    cleandf = df[['lead_time','avg_price_per_room','no_of_special_requests','arrival_date','arrival_month','market_segment_type','no_of_week_nights','booking_status']]
    categorical_cols = cleandf.select_dtypes(include='object').columns
    le = LabelEncoder()
    cleandf[categorical_cols] = cleandf[categorical_cols].apply(lambda col: le.fit_transform(col))
    one_hot_scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(cleandf), columns=cleandf.columns)
    dataset = one_hot_scaled_df
    # return one_hot_scaled_df
    return cleandf
@app.route('/train')
def trainModel():
    df = pd.read_csv(r'E:\George Brown College Semester 1 2\SEM 2\Full stack\Project\Deep-Learning-1\Hotel Reservations.csv')
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

    path_checkpoint = r"E:\George Brown College Semester 1 2\SEM 2\Full stack\Project\Deep-Learning-1\flask\training\cp.ckpt"
    directory_checkpoint = os.path.dirname(path_checkpoint)

    callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,
                                                 save_weights_only=True,
                                                 verbose=1)

    history = model.fit(x = X_train, y = y_train, validation_data = (X_val, y_val),  batch_size = 256, epochs = 100, callbacks=[callback])

    model.save(r'E:\George Brown College Semester 1 2\SEM 2\Full stack\Project\Deep-Learning-1\flask\saved_models\model1')
    print('Model Created and Saved')
    loaded_model = tf.keras.models.load_model(r'E:\George Brown College Semester 1 2\SEM 2\Full stack\Project\Deep-Learning-1\flask\saved_models\model1')
    print('Model loaded')
    print(loaded_model.summary())
    return { 'msg':'Model Trained'}

def modelInference(x,path = r'E:\George Brown College Semester 1 2\SEM 2\Full stack\Project\Deep-Learning-1\flask\saved_models\model1'):
    loaded_model = tf.keras.models.load_model(path)
    prediction = loaded_model.predict(np.array( [x,] ) )
    print(prediction,prediction.round())
    return prediction,prediction.round()

# trainModel()
# x = [5,	106.68,	1,	6,	11,	4,	3]
# modelInference(x)

@app.route('/')
def home():
    return render_template('reservation.html')

@app.route('/predict',  methods=['GET', 'POST'])
def predictForInput():
    lead_time = float(request.form.get("lead_time"))	
    avg_price_per_room = float(request.form.get('avg_price_per_room'))
    no_of_special_requests = float(request.form.get('special_requests'))
    date    =   (request.form.get('checkindate'))	
    dateArray = date.split('-')
    arrival_date = float(dateArray[2])
    arrival_month = float(dateArray[1])
    market_segment_type =	float(request.form.get('market_segment_type'))
    no_of_week_nights   =   float(request.form.get('no_of_week_nights'))

    x = [lead_time, avg_price_per_room,no_of_special_requests,arrival_date,arrival_month,market_segment_type,no_of_week_nights]
    pre,prediction = modelInference(x)
    print(prediction[0][0])
    res =''
    if int(prediction[0][0]==1):
        res = 'Not Cancel'
    else :
        res = 'Cancel'
    return render_template('result.html',result = [res,pre[0][0]])

# if __name__ == "__main__":
    # app.run(host='0.0.0.0', port='5000', debug=True)
