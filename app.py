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

train_data = pd.read_csv('Resources/train.csv')
test_data = pd.read_csv('Resources/test.csv')
train_data.head()

train_data_cleaned = pd.read_csv('AG_train-test.csv')
train_data_cleaned.head()

X_cleaned = train_data_cleaned[["Sex","Pclass","SibSp", "Age", "Fare", "Embarked"]]
y_cleaned = train_data_cleaned["Survived"].values.reshape(-1, 1)
target_names = ["Survived", "Not Survived"]
print(X_cleaned.shape, y_cleaned.shape)

X_cleaned = pd.get_dummies(X_cleaned, columns=["Sex"])
X_cleaned.head()

# Create the bins in which Data will be held
# Bins are 0 to 25, 25 to 50, 50 to 75, 75 to 100
bins = [0, 100, 200, 300, 400, 550]

# Create the names for the four bins
fare_group_names = ['Vey Low',"Low", 'Okay', 'High', 'Highest']

X_cleaned["Fare"] = pd.cut(X_cleaned["Fare"], bins, labels=fare_group_names)
X_cleaned.head()

X_cleaned = pd.get_dummies(X_cleaned, columns=["Fare"])
X_cleaned.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, random_state=42)

# Support vector machine linear classifier
from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(X_train, y_train)




predictions = model.predict(X_test)

newUser=[]
newUser.append([3,0,38.0,0,0,1,1,0,0,0,0])
print(X_test)
prediction = 0
prediction = model.predict(newUser)
prediction = prediction.tolist()
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
        userName = request.form["userName"]
        newUser.append(userName)
        userAge = request.form["userAge"]
        newUser.append(userAge)
        userTicket = request.form["userTicket"]
        newUser.append(userTicket)
        
        return redirect("/result", code=302)

    return render_template("form.html")


@app.route("/result")
def pals():

    if prediction[0] == 1:
        return render_template('result.html', prediction = "Survive" )
    else:
        return render_template('result.html', prediction = "Die" )

#Run the app. debug=True is essential to be able to rerun the server any time changes are saved to the Python file
if __name__ == "__main__":
    app.run(debug=True, port=5010)