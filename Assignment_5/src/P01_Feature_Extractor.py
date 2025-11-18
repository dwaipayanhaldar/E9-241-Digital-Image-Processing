import matplotlib.pyplot as plt

def plot_loss_curves(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Categorical Cross-Entropy)')
    plt.legend()
    plt.grid(True)
    plt.show()