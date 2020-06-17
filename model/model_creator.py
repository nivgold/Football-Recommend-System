import preprocessing.prepare_dataset as prepare
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import xgboost

leagues_datasets = prepare.preprocess()

leagues_models = {}

for league_name, dataset in leagues_datasets.items():
    X_train, y_train, X_test, y_test = dataset
    xgb_model = xgboost.XGBClassifier()

    optimization_dict = {'max_depth': [2, 4, 6],
                         'n_estimators': [50, 100, 200]}

    model = GridSearchCV(xgb_model, optimization_dict,
                         scoring='accuracy', verbose=1)

    model.fit(X_train, y_train)

    print(model.best_score_)

    # predictions
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    print(predictions)

    #eval
    test_accuracy = accuracy_score(y_test, predictions)
    print(f'League: {league_name}')
    print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))

    leagues_models[league_name] = test_accuracy

avg_acc = np.mean([acc for l_name, acc in leagues_models.items()])
print("AVG accuracy: %.2f%%" % (avg_acc * 100.0))