import matplotlib.pyplot as plt


def plot_loss(train_data_list, val_data_list, early_stopping_epoch, ax=None, y_lim=(-0.2, 10), figsize=(15, 7)):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    ax.plot(range(len(train_data_list)), train_data_list, label="Train Loss", color="blue")
    ax.plot(range(len(val_data_list)), val_data_list, label="Validation Loss", color="red")
    ax.axvline(x=early_stopping_epoch, linestyle="--", label=f"Early Stopping (Epoch : {early_stopping_epoch})",
               color="black")
    ax.set_ylim(y_lim)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Train-Validation Loss")
    ax.legend()

    return ax


def plot_accuracy(train_data_list, val_data_list, early_stopping_epoch, ax=None, y_lim=(-0.02, 1.02), figsize=(15, 7)):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    ax.plot(range(len(train_data_list)), train_data_list, label="Train Accuracy", color="blue")
    ax.plot(range(len(val_data_list)), val_data_list, label="Validation Accuracy", color="red")
    ax.axvline(x=early_stopping_epoch, linestyle="--", label=f"Early Stopping (Epoch : {early_stopping_epoch})",
               color="black")
    ax.set_ylim(y_lim)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title("Train-Validation Accuracy")
    ax.legend()

    return ax


def plot_precision(train_data_list, val_data_list, early_stopping_epoch, ax=None, y_lim=(-0.02, 1.02), figsize=(15, 7)):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    ax.plot(range(len(train_data_list)), train_data_list, label="Train Precision", color="blue")
    ax.plot(range(len(val_data_list)), val_data_list, label="Validation Precision", color="red")
    ax.axvline(x=early_stopping_epoch, linestyle="--", label=f"Early Stopping (Epoch : {early_stopping_epoch})",
               color="black")
    ax.set_ylim(y_lim)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Precision")
    ax.set_title("Train-Validation Precision")
    ax.legend()

    return ax


def plot_recall(train_data_list, val_data_list, early_stopping_epoch, ax=None, y_lim=(-0.02, 1.02), figsize=(15, 7)):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    ax.plot(range(len(train_data_list)), train_data_list, label="Train Recall", color="blue")
    ax.plot(range(len(val_data_list)), val_data_list, label="Validation Recall", color="red")
    ax.axvline(x=early_stopping_epoch, linestyle="--", label=f"Early Stopping (Epoch : {early_stopping_epoch})",
               color="black")
    ax.set_ylim(y_lim)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Recall")
    ax.set_title("Train-Validation Recall")
    ax.legend()

    return ax


def plot_f1(train_data_list, val_data_list, early_stopping_epoch, ax=None, y_lim=(-0.02, 1.02), figsize=(15, 7)):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    ax.plot(range(len(train_data_list)), train_data_list, label="Train F1 Score", color="blue")
    ax.plot(range(len(val_data_list)), val_data_list, label="Validation F1 Score", color="red")
    ax.axvline(x=early_stopping_epoch, linestyle="--", label=f"Early Stopping (Epoch : {early_stopping_epoch})",
               color="black")
    ax.set_ylim(y_lim)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("F1")
    ax.set_title("Train-Validation F1 Score")
    ax.legend()

    return ax


def plot_roc_curve(fpr, tpr, auc, ax=None, y_lim=(-0.02, 1.02), figsize=(8, 8)):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    ax.plot(fpr, tpr, label="ROC", color="red")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    ax.fill_between(fpr, 0, tpr, alpha=0.5, label=f"AUC : {auc}")
    ax.set_ylim(y_lim)
    ax.set_xlabel("Fall-Out")
    ax.set_ylabel("Recall")
    ax.set_title("ROC Curve")
    ax.legend()

    return ax