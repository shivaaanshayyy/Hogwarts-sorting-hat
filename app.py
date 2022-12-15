import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import OneHotEncoder

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    features = [x for x in request.form.values()]
    features=pd.DataFrame([features],columns=['Birthday','Best Hand','Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying'])
    features['Birthday'] = pd.to_datetime(features['Birthday'])
    # separate datetime into day, month, year features
    features['Birth_day'] = features['Birthday'].dt.day
    features['Birth_month'] = features['Birthday'].dt.month
    features['Birth_year'] = features['Birthday'].dt.year
    features = features.drop(columns=['Birthday'])
    print(features.columns)
    print(features.shape)


   
    ct = LabelEncoder()
    features['Best Hand']= ct.fit_transform(features["Best Hand"])
    
    #features1 = ct.fit(features)
    #features1 = features1.transform(features)
    #print(features1.shape)
    
    pred_features = np.array(features)
    #print(pred_features)
    #features=features.reshape(1,-1)
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "You Belong to {} Hogwarts House".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)