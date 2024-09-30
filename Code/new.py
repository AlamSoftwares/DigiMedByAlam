
#Training Using SVM Algorithm
model_SVC = SVC()
model_SVC.fit(X_train, y_train)
result_svc = model_SVC.predict(X_test)

#Training The Model Using Naive Bayes Algorithm
gnb = GaussianNB()
model_NB = gnb.fit(X_train, y_train)
result_NB = model_NB.predict(X_test)

#Training The Model Using RandomForestClassifier - Decision Tree
RFC = RandomForestClassifier(n_estimators=100, random_state=16)
model_RFC = RFC.fit(X_train, y_train)
result_RFC = RFC.predict(X_test)

#################################################
