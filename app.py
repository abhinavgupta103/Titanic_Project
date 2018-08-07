# import necessary libraries
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect)

from flask_sqlalchemy import SQLAlchemy

#################################################
# Flask Setup
#################################################
app = Flask(__name__)

import warnings
warnings.simplefilter('ignore')

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# read train and test csv files
train_data = pd.read_csv('Resources/train.csv')
test_data = pd.read_csv('Resources/test.csv')

# read cleaned csv from AG_train test
train_data_cleaned = pd.read_csv('AG_train-test.csv')

# select the columns will be used in testing
X_cleaned = train_data_cleaned[["Sex","Pclass","SibSp", "Age", "Fare", "Embarked"]]
y_cleaned = train_data_cleaned["Survived"].values.reshape(-1, 1)
target_names = ["Survived", "Not Survived"]

# get Sex collumn to be binary
X_cleaned = pd.get_dummies(X_cleaned, columns=["Sex"])


# Create the bins in which Data will be held Bins 
bins = [0, 100, 200, 300, 400, 550]

# Create the names for the four bins
fare_group_names = ['Vey Low',"Low", 'Okay', 'High', 'Highest']

X_cleaned["Fare"] = pd.cut(X_cleaned["Fare"], bins, labels=fare_group_names)

# get Fare column to be baniry
X_cleaned = pd.get_dummies(X_cleaned, columns=["Fare"])

# import test split and split data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, random_state=42)

# Support vector machine linear classifier
from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# import pickle to store model
import pickle
s = pickle.dumps(model)
model = pickle.loads(s)

# import joblib to load model
from sklearn.externals import joblib
joblib.dump(model, 'model.pkl') 
model = joblib.load('model.pkl') 

# run X_test to see predictions
predictions = model.predict(X_test)

# create new User list to store newUser data
newUser=[]
newUser.append([3,0,35,0,0,1,1,0,0,0,0])

# make prediction with new user data
prediction = model.predict(newUser)
print(prediction)


@app.route("/")
def index():
    predictions_dict = {

    }
    for x in predictions:
        count = 0
        if x == 0:
            predictions_dict["Prediction"] = "Not Survived"
        else:
            predictions_dict["Prediction"] = "Survived"    
    return render_template('index.html', predictions = predictions_dict )

@app.route("/send", methods=["GET", "POST"])
def send():
    if request.method == "POST":
        newUser.append([3,0,35,0,0,1,1,0,0,0,0])
        
        userEmbarked = request.form["userEmbarked"]
        userAge = request.form["userAge"]
        userTicket= request.form["userTicket"]
        userGender= request.form["userGender"]
        # userName[0] = request.form["userName"]
        # newUser.append(userName)
        # userAge = request.form["userAge"]
        # newUser.append(userAge)
        # userTicket = request.form["userTicket"]
        # newUser.append(userTicket)
        newUser[0][1]= int(userEmbarked)
        newUser[0][2]= int(userAge)
        newUser[0][6]= int(userTicket)
        
        if userGender == "male":
            newUser[0][4]= 1
            newUser[0][5]= 0
        else:
            newUser[0][4]= 0
            newUser[0][5]= 1

        prediction = model.predict(newUser)
        
        return redirect("/result", code=302)
        # return jsonify(newUser)

    return render_template("form.html")


@app.route("/result")
def pals():
    prediction = model.predict(newUser)

    if prediction[0] == 1:
        return render_template('result.html', prediction = "Survive", result_list = newUser )
    else:
        return render_template('result.html', prediction = "Die", result_list = newUser )

#Run the app. debug=True is essential to be able to rerun the server any time changes are saved to the Python file
if __name__ == "__main__":
    app.run(debug=True, port=5010)