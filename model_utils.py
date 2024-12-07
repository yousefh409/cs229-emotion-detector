import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, make_scorer, f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate

def get_metrics(train_preds, y_train, test_preds, y_test, train_probs, test_probs): 
    metrics = {}
    
    metrics['train_accuracy'] = accuracy_score(y_train, train_preds)
    metrics['train_precision'] = precision_score(y_train, train_preds, average="weighted", zero_division=0)
    metrics['train_recall'] = recall_score(y_train, train_preds, average="weighted", zero_division=0)
    metrics['train_auc'] = roc_auc_score(y_train, train_probs, multi_class='ovr', average="weighted")
    metrics['train_confusion_matrix'] = confusion_matrix(y_train, train_preds)
    metrics['train_f1'] = f1_score(y_train, train_preds, average="weighted", zero_division=0)

    metrics['test_accuracy'] = accuracy_score(y_test, test_preds)
    metrics['test_precision'] = precision_score(y_test, test_preds, average="weighted", zero_division=0)
    metrics['test_recall'] = recall_score(y_test, test_preds, average="weighted", zero_division=0)
    metrics['test_auc'] = roc_auc_score(y_test, test_probs, multi_class='ovr', average="weighted")
    metrics['test_f1'] = f1_score(y_test, test_preds, average="weighted", zero_division=0)
    metrics['test_confusion_matrix'] = confusion_matrix(y_test, test_preds)

    return metrics


def plot_model_comparison(model_name, models, test_accuracies, train_accuracies):
    x = np.arange(len(models)) 
    width = 0.35 

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, train_accuracies, width, label='Cross-Val Accuracy', color='blue')
    bars2 = ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy', color='orange')

    ax.set_ylim(0, 1.35)
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{model_name} Comparison: Cross-Val vs Test Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.show()



def plot_training_history(history):
    metrics = ['valence', 'arousal', 'attention', 'stress']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()  
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        train_acc = history.history[f'{metric}_acc']
        val_acc = history.history[f'val_{metric}_acc']
        
        ax.plot(train_acc, label=f'Training Accuracy ({metric})')
        ax.plot(val_acc, label=f'Validation Accuracy ({metric})')
        
        ax.set_title(f'{metric.capitalize()} Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def print_averages(metrics):
    train_accuracies = []
    train_precisions = []
    train_recalls = []
    train_aucs = []
    train_f1s = []

    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_aucs = []
    test_f1s = []

    for metric in metrics:
        train_accuracies.append(metric["train_accuracy"])
        train_precisions.append(metric["train_precision"])
        train_recalls.append(metric["train_recall"])
        train_aucs.append(metric["train_auc"])
        train_f1s.append(metric["train_f1"])

        test_accuracies.append(metric["test_accuracy"])
        test_precisions.append(metric["test_precision"])
        test_recalls.append(metric["test_recall"])
        test_aucs.append(metric["test_auc"])
        test_f1s.append(metric["test_f1"])

    
    print("Average Train Accuracy", np.mean(train_accuracies))
    print("Average Test Accuracy", np.mean(test_accuracies))
    print("Average Train Precision", np.mean(train_precisions))
    print("Average Test Precision", np.mean(test_precisions))
    print("Average Train Recall", np.mean(train_recalls))
    print("Average Test Recall", np.mean(test_recalls))
    print("Average Train AUC", np.mean(train_aucs))
    print("Average Test AUC", np.mean(test_aucs))
    print("Average Train F1", np.mean(train_f1s))
    print("Average Test F1", np.mean(test_f1s))

def plot_confusion_matricies(cms, targets):
    num_outputs = 4
    fig, axes = plt.subplots(1, num_outputs, figsize=(5 * num_outputs, 5))
    for i, (cm, target) in enumerate(zip(cms, targets)):
        ConfusionMatrixDisplay(cm).plot(ax=axes[i])
        # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[i])
        axes[i].set_title(f"Confusion Matrix: {target.capitalize()}")
    plt.tight_layout()
    plt.show()


def confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    return confusion_matrix(y, y_pred)

def get_cv_metrics(model, X, y, X_test, y_test, cv):
    cms = []

    # return_train_score=True
    scores = cross_validate(model, X, y, cv=cv, scoring={"acccuracy": "accuracy", 
                                                         "f1_weighted": "f1_weighted",
                                                         "precision_weighted": "precision_weighted",
                                                         "recall_weighted": "recall_weighted",
                                                         "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
                                                         "cm": lambda clf, X, y: cms.append(confusion_matrix_scorer(clf, X, y)) or 1})
    
    metrics = {}

    model.fit(X, y)


    metrics['crossval_accuracy'] = np.mean(scores["test_acccuracy"])
    metrics['crossval_f1'] = np.mean(scores["test_f1_weighted"])
    metrics['crossval_precision'] = np.mean(scores["test_precision_weighted"])
    metrics['crossval_recall'] = np.mean(scores["test_recall_weighted"])
    metrics['crossval_auc'] = np.mean(scores["test_roc_auc_ovr_weighted"])
    metrics['crossval_cm'] = np.mean(cms, axis=0)

    metrics['crossval_accuracy_sd'] = np.std(scores["test_acccuracy"])
    metrics['crossval_f1_sd'] = np.std(scores["test_f1_weighted"])
    metrics['crossval_precision_sd'] = np.std(scores["test_precision_weighted"])
    metrics['crossval_recall_sd'] = np.std(scores["test_recall_weighted"])
    metrics['crossval_auc_sd'] = np.std(scores["test_roc_auc_ovr_weighted"])

    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)

    metrics['test_accuracy'] = accuracy_score(y_test, test_preds)
    metrics['test_precision'] = precision_score(y_test, test_preds, average="weighted", zero_division=0)
    metrics['test_recall'] = recall_score(y_test, test_preds, average="weighted", zero_division=0)
    metrics['test_f1'] = f1_score(y_test, test_preds, average="weighted", zero_division=0)

    metrics['test_cm'] =  confusion_matrix(y_test, test_preds)
    metrics['test_auc'] = roc_auc_score(y_test, test_probs, multi_class='ovr', average="weighted")


    return metrics

def print_cv_metrics(metrics):
    crossval_accs, crossval_acc_stds = [], []
    crossval_f1s, crossval_f1_stds = [], []
    crossval_precs, crossval_prec_stds = [], []
    crossval_recalls, crossval_recall_stds = [], []
    crossval_aucs, crossval_auc_stds = [], []

    test_accs = []
    test_f1s =  []
    test_precs = []
    test_recalls = []
    test_aucs = []

    for metric in metrics:
        crossval_accs.append(metric["crossval_accuracy"])
        crossval_acc_stds.append(metric["crossval_accuracy_sd"])
        crossval_f1s.append(metric["crossval_f1"])
        crossval_f1_stds.append(metric["crossval_f1_sd"])
        crossval_precs.append(metric["crossval_precision"])
        crossval_prec_stds.append(metric["crossval_precision_sd"])
        crossval_recalls.append(metric["crossval_recall"])
        crossval_recall_stds.append(metric["crossval_recall_sd"])
        crossval_aucs.append(metric["crossval_auc"])
        crossval_auc_stds.append(metric["crossval_auc_sd"])

        test_accs.append(metric["test_accuracy"])
        test_f1s.append(metric["test_f1"])
        test_precs.append(metric["test_precision"])
        test_recalls.append(metric["test_recall"])
        test_aucs.append(metric["test_auc"])

    print("Cross-Val Accuracy:", np.mean(crossval_accs), "+=", np.mean(crossval_acc_stds))    
    print("Cross-Val F1:", np.mean(crossval_f1s), "+=", np.mean(crossval_f1_stds))    
    print("Cross-Val Precision:", np.mean(crossval_precs), "+=", np.mean(crossval_prec_stds))    
    print("Cross-Val Recall:", np.mean(crossval_recalls), "+=", np.mean(crossval_recall_stds))    
    print("Cross-Val AUC:", np.mean(crossval_aucs), "+=", np.mean(crossval_auc_stds))    

    print("Test Accuracy:", np.mean(test_accs))    
    print("Test F1:", np.mean(test_f1s))    
    print("Test Precision:", np.mean(test_precs))    
    print("Test Recall:", np.mean(test_recalls))    
    print("Test AUC:", np.mean(test_aucs))    



def get_grid_search_cv_metrics(grid_search, X_test, y_test):
    cms = []
    metrics = {}

    metrics['crossval_accuracy'] = grid_search.cv_results_["mean_test_accuracy"][grid_search.best_index_]
    metrics['crossval_f1'] = grid_search.cv_results_["mean_test_f1_weighted"][grid_search.best_index_]
    metrics['crossval_precision'] =  grid_search.cv_results_["mean_test_precision_weighted"][grid_search.best_index_]
    metrics['crossval_recall'] = grid_search.cv_results_["mean_test_recall_weighted"][grid_search.best_index_]
    metrics['crossval_auc'] = grid_search.cv_results_["mean_test_roc_auc_ovr_weighted"][grid_search.best_index_]


    test_preds = grid_search.best_estimator_.predict(X_test)
    test_probs = grid_search.best_estimator_.predict_proba(X_test)

    metrics['test_accuracy'] = accuracy_score(y_test, test_preds)
    metrics['test_precision'] = precision_score(y_test, test_preds, average="weighted", zero_division=0)
    metrics['test_recall'] = recall_score(y_test, test_preds, average="weighted", zero_division=0)
    metrics['test_f1'] = f1_score(y_test, test_preds, average="weighted", zero_division=0)

    metrics['test_cm'] =  confusion_matrix(y_test, test_preds)
    metrics['test_auc'] = roc_auc_score(y_test, test_probs, multi_class='ovr', average="weighted")


    metrics['crossval_accuracy_sd'] =  grid_search.cv_results_["std_test_accuracy"][grid_search.best_index_]
    metrics['crossval_f1_sd'] =  grid_search.cv_results_["std_test_f1_weighted"][grid_search.best_index_]
    metrics['crossval_precision_sd'] =  grid_search.cv_results_["std_test_precision_weighted"][grid_search.best_index_]
    metrics['crossval_recall_sd'] = grid_search.cv_results_["std_test_recall_weighted"][grid_search.best_index_]
    metrics['crossval_auc_sd'] =  grid_search.cv_results_["std_test_roc_auc_ovr_weighted"][grid_search.best_index_]

    return metrics


from sklearn.model_selection import GridSearchCV

def get_grid_search(model, param_grid):
    return GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=7,               
        verbose=2,          
        n_jobs=-1,
        refit="accuracy",
        scoring={"accuracy": "accuracy", 
                 "f1_weighted": "f1_weighted",
                 "precision_weighted": "precision_weighted",
                 "recall_weighted": "recall_weighted",
                 "roc_auc_ovr_weighted": "roc_auc_ovr_weighted"}           
    )


def get_test_metrics(test_preds, y_test, test_probs): 
    metrics = {}

    metrics['test_accuracy'] = accuracy_score(y_test, test_preds)
    metrics['test_precision'] = precision_score(y_test, test_preds, average="weighted", zero_division=0)
    metrics['test_recall'] = recall_score(y_test, test_preds, average="weighted", zero_division=0)
    metrics['test_auc'] = roc_auc_score(y_test, test_probs, multi_class='ovr', average="weighted")
    metrics['test_confusion_matrix'] = confusion_matrix(y_test, test_preds)

    return metrics