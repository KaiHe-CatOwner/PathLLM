import numpy as np

def clip_path_map(path):
    if path=="pathclip-base":
        return "load_weights/pathclip/pathclip-base.pt"
    
    if path=="conch":
        return "load_weights/conch/pytorch_model.bin"


def my_compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)
    
    mask = labels != -1
    correct = (labels == predictions) & mask
    accuracy = np.mean(correct)

    return {
        'accuracy': accuracy,
    }