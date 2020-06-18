import preprocessing.prepare_dataset as prepare
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import itertools
import numpy as np
import xgboost
from xgboost import plot_importance
from matplotlib import pyplot

each_league_dataset = prepare.preprocess_each_league()
all_leagues_dataset = prepare.preprocess_all_leagues()


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
    pyplot.title("Confusion Matrix of State-of-The-Art XGBoost Model")
    pyplot.tight_layout()
    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')

    pyplot.show()


def xgb_all_leagues():
    X_train, y_train, X_test, y_test = all_leagues_dataset

    # number of epochs
    epochs = 1500

    model = xgboost.XGBClassifier(
        learning_rate=0.01,
        objective='multi:softmax',
        n_estimators=epochs,
        max_depth=3
    )

    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, early_stopping_rounds=epochs * 0.005, eval_metric=["mlogloss"],
              eval_set=eval_set, verbose=False)

    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['mlogloss'])
    x_axis = range(0, epochs)

    # plot classification error
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
    ax.legend()
    pyplot.ylabel('Classification Error')
    pyplot.title('XGBoost Classification Error')
    pyplot.show()

    # get train predictions
    y_train_pred = model.predict(X_train)
    train_predictions = [round(val) for val in y_train_pred]

    # get test predictions
    y_pred = model.predict(X_test)
    predictions = [round(val) for val in y_pred]

    # calculate train and test accuracies
    accuracy = accuracy_score(y_test, predictions)
    print("TRAIN acc: %.2f%%" % (accuracy_score(y_train, train_predictions) * 100.0))
    print("TEST acc: %.2f%%" % (accuracy * 100.0))

    # plotting confusion matrix
    plot_cm(y_test, predictions)

    # print precision recall f1 support table
    print_classification_table(y_test, predictions)

    # plot feature importance
    plot_importance(model)
    pyplot.show()


def xgb_each_league():
    leagues_models = {}

    for league_name, dataset in each_league_dataset.items():
        X_train, y_train, X_test, y_test = dataset

        epochs = 1000

        model = xgboost.XGBClassifier(
            objective='multi:softmax',
            n_estimators=epochs,
            max_depth=3
        )

        eval_set = [(X_train, y_train), (X_test, y_test)]
        model.fit(X_train, y_train, early_stopping_rounds=epochs * 0.1, eval_metric=["merror", "mlogloss"],
                  eval_set=eval_set, verbose=False)

        y_train_pred = model.predict(X_train)
        train_predictions = [round(val) for val in y_train_pred]

        y_pred = model.predict(X_test)
        predictions = [round(val) for val in y_pred]

        train_accuracy = accuracy_score(y_train, train_predictions) * 100.0
        test_accuracy = accuracy_score(y_test, predictions) * 100.0
        print(f'\n\n{league_name}')
        print("TRAIN acc: %.2f%%" % (train_accuracy))
        print("TEST acc: %.2f%%" % (test_accuracy))

        leagues_models[league_name] = (train_accuracy, test_accuracy)

    avg_train_acc = np.mean([train_accuracy for l_name, (train_accuracy, test_accuracy) in leagues_models.items()])
    avg_test_acc = np.mean([test_accuracy for l_name, (train_accuracy, test_accuracy) in leagues_models.items()])
    print("Average Train Accuracy: %.2f%%" % (avg_train_acc))
    print("Average Test Accuracy: %.2f%%" % (avg_test_acc))


def main():
    xgb_all_leagues()


if __name__ == '__main__':
    main()
