import numpy as np
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


def get_metrics(train_preds, y_train, test_preds, y_test): 

    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)

    return train_accuracy, test_accuracy


def plot_model_comparison(model_name, models, test_accuracies, train_accuracies):
    """
    Plots a bar graph comparing test and train accuracies for each model.
    
    Args:
        models (list): List of model names.
        test_accuracies (list): List of test accuracy values for the models.
        train_accuracies (list): List of train accuracy values for the models.
    """
    x = np.arange(len(models))  # Label locations
    width = 0.35  # Bar width

    # Create the bar plot
    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, train_accuracies, width, label='Train Accuracy', color='blue')
    bars2 = ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy', color='orange')

    # Add text for labels, title, and axes ticks
    ax.set_ylim(0, 1.35)
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{model_name} Comparison: Train vs Test Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    # Adding numerical labels above each bar
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Offset text by 3 points
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.show()



def plot_training_history(history):
    """
    Plots training and validation accuracy for each output from the model's history.
    
    Args:
        history (tf.keras.callbacks.History): The history object returned by model.fit().
    """
    metrics = ['valence', 'arousal', 'attention', 'stress']
    
    # Create subplots for each output
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()  # Flatten 2D array to easily iterate
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Extract train and validation accuracies
        train_acc = history.history[f'{metric}_acc']
        val_acc = history.history[f'val_{metric}_acc']
        
        # Plot accuracies
        ax.plot(train_acc, label=f'Training Accuracy ({metric})')
        ax.plot(val_acc, label=f'Validation Accuracy ({metric})')
        
        # Add titles and labels
        ax.set_title(f'{metric.capitalize()} Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
