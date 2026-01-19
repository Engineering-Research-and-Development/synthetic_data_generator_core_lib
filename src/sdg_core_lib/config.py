import os


def get_hyperparameters() -> dict:
    return {
        "epochs": os.environ.get("EPOCHS"),
        "learning_rate": os.environ.get("LEARNING_RATE"),
        "batch_size": os.environ.get("BATCH_SIZE"),
    }
