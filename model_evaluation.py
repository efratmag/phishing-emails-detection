"""
Functions for models evaluation
"""

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