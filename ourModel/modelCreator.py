import ourModel.prepare_dataset as prepare
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import itertools
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance
from matplotlib import pyplot
import numpy as np
import xgboost


def get_labeled_class(y_test, predictions):
    y_test_labels = ["Draw" if val == 0 else "Win" if val == 1 else "Loss" for val in y_test]
    predictions_labels = ["Draw" if val == 0 else "Win" if val == 1 else "Loss" for val in predictions]
    return y_test_labels, predictions_labels


def print_classification_table(y_test, predictions):
    target_names = ["Draw", "Win", "Loss"]
    y_test, predictions = get_labeled_class(y_test, predictions)
    print(classification_report(y_test, predictions, target_names=target_names))


def plot_cm(y_test, predictions, normalize=True):
    labels = ["Draw", "Win", "Loss"]
    y_test, predictions = get_labeled_class(y_test, predictions)
    cm = confusion_matrix(y_test, predictions, labels=labels)

    if normalize == True:
        cm = cm.astype('float') / cm.sum()

    pyplot.imshow(cm, interpolation='nearest', cmap=pyplot.cm.Greens)
    pyplot.colorbar()
    tick_marks = np.arange(len(labels))
    pyplot.xticks(tick_marks, labels, rotation=45)
    pyplot.yticks(tick_marks, labels)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pyplot.text(j, i, round(cm[i, j], 2),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    pyplot.title("Confusion Matrix of Our XGBoost Model")
    pyplot.tight_layout()
    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')

    pyplot.show()


leagues_datasets = prepare.preprocess()

leagues_models = {}

X_train, y_train, X_test, y_test = leagues_datasets
epochs = 200
model = xgboost.XGBClassifier(
    objective='multi:softmax',
    learning_rate=0.1,
    n_estimators=epochs,
    max_depth=3
)

eval_set = [(X_train, y_train), (X_test, y_test)]

model.fit(X_train, y_train, early_stopping_rounds=epochs * 0.1, eval_metric=["merror", "mlogloss"], eval_set=eval_set,
          verbose=False)

# predictions
y_train_pred = model.predict(X_train)
train_predictions = [round(value) for value in y_train_pred]
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# eval train:
train_accuracy = accuracy_score(y_train, train_predictions)
print("TRAIN Accuracy: %.2f%%" % (train_accuracy * 100.0))

# eval test:
test_accuracy = accuracy_score(y_test, predictions)
print("TEST Accuracy: %.2f%%" % (test_accuracy * 100.0))

# results for plots:
results = model.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)

# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()

# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.show()

# feature importance:
plot_importance(model)
pyplot.show()

# confusion matrix:
plot_cm(y_test, predictions)

# print precision recall f1 support table
print_classification_table(y_test, predictions)
