import numpy as np

import gzip




def load_images(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        images = data.reshape(-1, 28, 28)
    return images
def load_labels(path):
    with open(path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels

