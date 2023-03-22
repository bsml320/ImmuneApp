import numpy as np

def scoring(models, data):
    probas_ = [np.transpose(model.predict(data))[0] for model in models]
    probas_ = [np.mean(scores) for scores in zip(*probas_)]
    return probas_  