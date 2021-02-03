"""
Functions for models evaluation
"""

import pandas as pd
import numpy as np
import seaborn as sns
import timeit
import pickle
import sys

from sklearn.exceptions import NotFittedError
from sklearn.metrics import make_scorer,f1_score
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score,roc_curve, \
                            confusion_matrix, recall_score, precision_score, \
                            precision_recall_curve,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier


import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'f1_score': make_scorer(f1_score)}

classifiers = {
    'log_model': make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000)),
    'svc_model': make_pipeline(StandardScaler(), SVC()),
    'dtr_model': DecisionTreeClassifier(),
    'rfc_model': RandomForestClassifier(),
    'xgb_model': XGBClassifier(eval_metric='error'),
    'adb_model': AdaBoostClassifier(n_estimators=100)
}

# Define the models evaluation function
def choose_model(classifiers, X, y, folds, metric):
    '''
    classifiers: dictionary of the classifiers to compare
    X : data set features
    y : data set target
    folds : number of cross-validation folds
    metric: scorer on which the selection is made

    '''
    models_scores = np.zeros((len(scoring),len(classifiers)))
    for model_name, model in classifiers.items():
        clf = cross_validate(model, X, y)

    # Perform cross-validation to each machine learning classifier
    log = cross_validate(log_model, X, y, cv=folds, scoring=scoring)
    svc = cross_validate(svc_model, X, y, cv=folds, scoring=scoring)
    dtr = cross_validate(dtr_model, X, y, cv=folds, scoring=scoring)
    rfc = cross_validate(rfc_model, X, y, cv=folds, scoring=scoring)
    gnb = cross_validate(gnb_model, X, y, cv=folds, scoring=scoring)
    xgb = cross_validate(xgb_model, X, y, cv=folds, scoring=scoring)
    adb = cross_validate(adb_model, X, y, cv=folds, scoring=scoring)

    # Create a data frame with the models perfoamnce metrics scores
    models_scores_table = pd.DataFrame({'Logistic Regression': [log['test_accuracy'].mean(),
                                                                log['test_precision'].mean(),
                                                                log['test_recall'].mean(),
                                                                log['test_f1_score'].mean()],

                                        'Support Vector Classifier': [svc['test_accuracy'].mean(),
                                                                      svc['test_precision'].mean(),
                                                                      svc['test_recall'].mean(),
                                                                      svc['test_f1_score'].mean()],

                                        'Decision Tree': [dtr['test_accuracy'].mean(),
                                                          dtr['test_precision'].mean(),
                                                          dtr['test_recall'].mean(),
                                                          dtr['test_f1_score'].mean()],

                                        'Random Forest': [rfc['test_accuracy'].mean(),
                                                          rfc['test_precision'].mean(),
                                                          rfc['test_recall'].mean(),
                                                          rfc['test_f1_score'].mean()],

                                        'Gaussian Naive Bayes': [gnb['test_accuracy'].mean(),
                                                                 gnb['test_precision'].mean(),
                                                                 gnb['test_recall'].mean(),
                                                                 gnb['test_f1_score'].mean()],

                                        'XGBoost Classifier': [xgb['test_accuracy'].mean(),
                                                               xgb['test_precision'].mean(),
                                                               xgb['test_recall'].mean(),
                                                               xgb['test_f1_score'].mean()],

                                        'AdABoost Classifier': [adb['test_accuracy'].mean(),
                                                                adb['test_precision'].mean(),
                                                                adb['test_recall'].mean(),
                                                                adb['test_f1_score'].mean()]},

                                       index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])

    # Add 'Best Score' column
    models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)

    # Return models performance metrics scores data frame
    return (models_scores_table)


# Run models_evaluation function
models_evaluation(X, y, 5)

# define a function to print accuracy metrics
def print_accuracy_metrics(Input,Output):
    print("Recall:", recall_score(Input, Output))
    print("Log Loss:", log_loss(Input, Output))
    print("Precision:", precision_score(Input, Output))
    print("Accurcay:", accuracy_score(Input, Output))
    print("AUC: ", roc_auc_score(Input, Output))
    print("F1 Score:", f1_score(Input, Output))
    confusion_matrix_value = confusion_matrix(Input,Output)
    print('Confusion matrix:\n', confusion_matrix_value)
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame( confusion_matrix_value), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')


def confusion_plot(matrix, labels=None):
    """ Display binary confusion matrix as a Seaborn heatmap """

    labels = labels if labels else ['Negative (0)', 'Positive (1)']

    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.heatmap(data=matrix, cmap='Blues', annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('PREDICTED')
    ax.set_ylabel('ACTUAL')
    ax.set_title('Confusion Matrix')
    plt.close()

    return fig

# defined a function to print cross validation score
scoring = {'recall' : make_scorer(recall_score)}
def cross_validation_metrics(log_reg, X, y):
    log_reg_score = cross_val_score(log_reg, X,y,cv=5,scoring='recall')
    print('Logistic Regression Cross Validation Score(Recall): ', round(log_reg_score.mean() * 100, 2)
          .astype(str) + '%')

# function to draw ROC curve
def roc_plot(y_true, y_probs, label, compare=False, ax=None):
    """ Plot Receiver Operating Characteristic (ROC) curve
        Set `compare=True` to use this function to compare classifiers. """

    fpr, tpr, thresh = roc_curve(y_true, y_probs,
                                 drop_intermediate=False)
    auc = round(roc_auc_score(y_true, y_probs), 2)

    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1)
    label = ' '.join([label, f'({auc})']) if compare else None
    sns.lineplot(x=fpr, y=tpr, ax=axis, label=label)

    if compare:
        axis.legend(title='Classifier (AUC)', loc='lower right')
    else:
        axis.text(0.72, 0.05, f'AUC = {auc}', fontsize=12,
                  bbox=dict(facecolor='green', alpha=0.4, pad=5))

        # Plot No-Info classifier
        axis.fill_between(fpr, fpr, tpr, alpha=0.3, edgecolor='g',
                          linestyle='--', linewidth=2)

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title('ROC Curve')
    axis.set_xlabel('False Positive Rate [FPR]\n(1 - Specificity)')
    axis.set_ylabel('True Positive Rate [TPR]\n(Sensitivity or Recall)')

    plt.close()

    return axis if ax else fig


# function to draw precision recall plot
def precision_recall_plot(y_true, y_probs, label, compare=False, ax=None):
    """ Plot Precision-Recall curve.
        Set `compare=True` to use this function to compare classifiers. """

    p, r, thresh = precision_recall_curve(y_true, y_probs)
    p, r, thresh = list(p), list(r), list(thresh)
    p.pop()
    r.pop()

    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1)

    if compare:
        sns.lineplot(r, p, ax=axis, label=label)
        axis.set_xlabel('Recall')
        axis.set_ylabel('Precision')
        axis.legend(loc='lower left')
    else:
        sns.lineplot(thresh, p, label='Precision', ax=axis)
        axis.set_xlabel('Threshold')
        axis.set_ylabel('Precision')
        axis.legend(loc='lower left')

        axis_twin = axis.twinx()
        sns.lineplot(thresh, r, color='limegreen', label='Recall', ax=axis_twin)
        axis_twin.set_ylabel('Recall')
        axis_twin.set_ylim(0, 1)
        axis_twin.legend(bbox_to_anchor=(0.24, 0.18))

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title('Precision Vs Recall')

    plt.close()

    return axis if ax else fig


def model_memory_size(clf):
    return sys.getsizeof(pickle.dumps(clf))


# function for plotting feature importance
def feature_importance_plot(importances, feature_labels, ax=None):
    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1, figsize=(5, 10))
    sns.barplot(x=importances, y=feature_labels, ax=axis)
    axis.set_title('Feature Importance Measures')

    plt.close()

    return axis if ax else fig


def train_clf(clf, x_train, y_train, sample_weight=None, refit=False):
    train_time = 0

    try:
        if refit:
            raise NotFittedError
        y_pred_train = clf.predict(x_train)
    except NotFittedError:
        start = timeit.default_timer()

        if sample_weight is not None:
            clf.fit(x_train, y_train, sample_weight=sample_weight)
        else:
            clf.fit(x_train, y_train)

        end = timeit.default_timer()
        train_time = end - start

        y_pred_train = clf.predict(x_train)

    train_acc = accuracy_score(y_train, y_pred_train)
    return clf, y_pred_train, train_acc, train_time


def report(clf, x_train, y_train, x_test, y_test, display_scores=[],
           sample_weight=None, refit=False, importance_plot=False,
           confusion_labels=None, feature_labels=None, verbose=True):
    """ Trains the passed classifier if not already trained and reports
        various metrics of the trained classifier """

    dump = dict()

    # Train if not already trained
    clf, train_predictions, \
    train_acc, train_time = train_clf(clf, x_train, y_train,
                                      sample_weight=sample_weight,
                                      refit=refit)
    # Testing
    start = timeit.default_timer()
    test_predictions = clf.predict(x_test)
    end = timeit.default_timer()
    test_time = end - start

    test_acc = accuracy_score(y_test, test_predictions)
    y_probs = clf.predict_proba(x_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_probs)

    # Additional scores
    scores_dict = dict()
    for func in display_scores:
        scores_dict[func.__name__] = [func(y_train, train_predictions),
                                      func(y_test, test_predictions)]

    # Model Memory
    model_mem = round(model_memory_size(clf) / 1024, 2)

    print(clf)
    print("\n=============================> TRAIN-TEST DETAILS <======================================")

    # Metrics
    print(f"Train Size: {x_train.shape[0]} samples")
    print(f" Test Size: {x_test.shape[0]} samples")
    print("---------------------------------------------")
    print(f"Training Time: {round(train_time, 3)} seconds")
    print(f" Testing Time: {round(test_time, 3)} seconds")
    print("---------------------------------------------")
    print("Train Accuracy: ", train_acc)
    print(" Test Accuracy: ", test_acc)
    print("---------------------------------------------")

    if display_scores:
        for k, v in scores_dict.items():
            score_name = ' '.join(map(lambda x: x.title(), k.split('_')))
            print(f'Train {score_name}: ', v[0])
            print(f' Test {score_name}: ', v[1])
            print()
        print("---------------------------------------------")

    print(" Area Under ROC (test): ", roc_auc)
    print("---------------------------------------------")
    print(f"Model Memory Size: {model_mem} kB")
    print("\n=============================> CLASSIFICATION REPORT <===================================")

    # Classification Report
    clf_rep = classification_report(y_test, test_predictions, output_dict=True)

    print(classification_report(y_test, test_predictions,
                                target_names=confusion_labels))

    if verbose:
        print("\n================================> CONFUSION MATRIX <=====================================")

        # Confusion Matrix HeatMap
        sns.display(confusion_plot(confusion_matrix(y_test, test_predictions),
                               labels=confusion_labels))
        print("\n=======================================> PLOTS <=========================================")

        # Variable importance plot
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
        roc_axes = axes[0, 0]
        pr_axes = axes[0, 1]
        importances = None

        if importance_plot:
            if not feature_labels:
                raise RuntimeError("'feature_labels' argument not passed "
                                   "when 'importance_plot' is True")

            try:
                importances = pd.Series(clf.feature_importances_,
                                        index=feature_labels) \
                    .sort_values(ascending=False)
            except AttributeError:
                try:
                    importances = pd.Series(clf.coef_.ravel(),
                                            index=feature_labels) \
                        .sort_values(ascending=False)
                except AttributeError:
                    pass

            if importances is not None:
                # Modifying grid
                grid_spec = axes[0, 0].get_gridspec()
                for ax in axes[:, 0]:
                    ax.remove()  # remove first column axes
                large_axs = fig.add_subplot(grid_spec[0:, 0])

                # Plot importance curve
                feature_importance_plot(importances=importances.values,
                                        feature_labels=importances.index,
                                        ax=large_axs)
                large_axs.axvline(x=0)

                # Axis for ROC and PR curve
                roc_axes = axes[0, 1]
                pr_axes = axes[1, 1]
            else:
                # remove second row axes
                for ax in axes[1, :]:
                    ax.remove()
        else:
            # remove second row axes
            for ax in axes[1, :]:
                ax.remove()

        # ROC and Precision-Recall curves
        clf_name = clf.__class__.__name__
        roc_plot(y_test, y_probs, clf_name, ax=roc_axes)
        precision_recall_plot(y_test, y_probs, clf_name, ax=pr_axes)

        fig.subplots_adjust(wspace=5)
        fig.tight_layout()
        sns.display(fig)

    # Dump to report_dict
    dump = dict(clf=clf, accuracy=[train_acc, test_acc], **scores_dict,
                train_time=train_time, train_predictions=train_predictions,
                test_time=test_time, test_predictions=test_predictions,
                test_probs=y_probs, report=clf_rep, roc_auc=roc_auc,
                model_memory=model_mem)

    return clf, dump
