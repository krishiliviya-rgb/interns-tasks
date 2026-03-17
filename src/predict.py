import numpy as np

def make_prediction(model, year, month, day):
    """
    Make prediction using trained model.
    """

    input_data = np.array([[year, month, day]])
    prediction = model.predict(input_data)

    return prediction[0]