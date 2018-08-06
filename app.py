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

train_data = train_data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
test_data = test_data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

X = train_data[["Sex","Pclass","SibSp", "Age", "Fare"]]
y = train_data["Survived"].values.reshape(-1, 1)
target_names = ["Survived", "Not Survived"]

X = pd.get_dummies(X, columns=["Sex"])

# Bins are 0 to 25, 25 to 50, 50 to 75, 75 to 100
bins = [0, 100, 200, 300, 400, 550]

# Create the names for the four bins
fare_group_names = ['Extra Economy Class',"Economy Class", 'Middle Class', 'Business Class', 'Extra Business Class']

X["Fare"] = pd.cut(X["Fare"], bins, labels=fare_group_names)

X = pd.get_dummies(X, columns=["Fare"])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Support vector machine linear classifier
from sklearn.svm import SVC 
model = SVC(kernel='linear')
model.fit(X_train, y_train)

model.score(X_test, y_test)

# Calculate classification report
from sklearn.metrics import classification_report


predictions = model.predict(X_test)

newUser=[]
newUser.append([25,0,1,0,1,0,0,0,0,0])
newUser_shaped = np.reshape(newUser, (-1, 1))

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
    return render_template('result.html', prediction = prediction )

#Run the app. debug=True is essential to be able to rerun the server any time changes are saved to the Python file
if __name__ == "__main__":
    app.run(debug=True, port=5010)