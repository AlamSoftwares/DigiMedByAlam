# Required libraries
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Reading Training Data
data_train = pd.read_csv("Dataset/Training.csv").dropna(axis=1)

#Converting Prognosis Object part into numerical form using sklearn LabelEncoder()
le = LabelEncoder()
detected = le.fit_transform(data_train["prognosis"])

#Splitting Data For Training And Testing
X = data_train.iloc[:, :-1]
y = data_train.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Implementing K-Fold Cross Validation, K=12
def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))


model_set = {
    "SVC": SVC(),
    "Gaussian NB": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=16)
}
for i in model_set:
    current_model = model_set[i]
    scores = cross_val_score(current_model, X, y, cv=12, n_jobs=-1, scoring=cv_scoring)


#Reading Test Data
data_test = pd.read_csv("Dataset/Testing.csv").dropna(axis=1)

#Selct Test Data
test_X = data_test.iloc[:, :-1]
test_Y = data_test.iloc[:, -1]

#Training Using SVM Algorithm
main_model_SVC = SVC()
main_model_SVC.fit(X, y)

#Training The Model Using Naive Bayes Algorithm
main_gnb = GaussianNB()
main_model_NB = main_gnb.fit(X, y)

#Training The Model Using RandomForestClassifier - Decision Tree Algorithm
main_RFC = RandomForestClassifier(n_estimators=100, random_state=16)
main_model_RFC = main_RFC.fit(X, y)



##########################################################################################################################


##GUI For App

#Required Libraries
from flask import Flask, render_template, request, send_from_directory

symptoms = X.columns.values

# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = "".join(value)
    symptom_index[symptom] = index

data_dict = {
    "symptom_index":symptom_index,
    "detection_classes":le.classes_
}

#Function For Disease Detection
def detect_Disease(symptoms):
    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
        
    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)

    #Using Models For Detections As Per User Given Symptoms
    SVM_detection = main_model_SVC.predict(input_data)[0]
    NB_detection = main_model_NB.predict(input_data)[0]
    RFC_detection = main_model_RFC.predict(input_data)[0]
    
    # making final detection by taking mode of all detection from all algorithms
    import statistics
    final_result = statistics.mode([RFC_detection, NB_detection, SVM_detection])
    detected_result = {
        "RandomForestClassifier Detected": RFC_detection,
        "Naive Bayes Classifier Detected": str(NB_detection),
        "SVM Classifier Detected": SVM_detection,
        "Thus You Have": final_result
    }
    RandomForestClassifier = RFC_detection
    NaiveBayes = str(NB_detection)
    SVM = SVM_detection
    fin_det = final_result
    return RandomForestClassifier,NaiveBayes,SVM,fin_det


app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():    
    result_RFC="None"
    result_NB="None"
    result_SVM="None"
    fin = "None"
    pat_name = ""
    result = {}
    #Form Input From Checkboxes
    if request.method=='POST':
        symptoms_values = np.array(request.form.getlist('sympt'))
        result_RFC, result_NB, result_SVM, fin = detect_Disease(symptoms_values)
        pat_name = request.form.get("patient_name")
        
    return render_template('index.html', res_RFC=result_RFC, res_NB= result_NB, res_SVM=result_SVM,fin=fin,name=pat_name)

@app.route('/files/Input.txt')
def serve_file(filename):
    # Ensure the directory is correct
    return send_from_directory('Code/Templates', filename)

if __name__ == '__main__':
    app.run(debug=True)
