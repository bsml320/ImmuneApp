def get_highest_binding(predictions):
    if 'Strong' in predictions.values:
        return 'Strong'
    elif 'Weak' in predictions.values:
        return 'Weak'
    else:
        return 'Non-binding' 