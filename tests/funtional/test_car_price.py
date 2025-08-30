import os
import json

def test_ml_page(test_client):
    """
    Given a Flask application configured for testing


    """
    models = os.listdir("models") 

    for model_name in models:
        response = test_client.get(f'/{model_name}')


        assert b'Car Price Prediction' in response.data


def test_ml_predict(test_client):
    """
    Given a Flask application configured for testing
    When the '/<model_name>' route is posted to (for each model in models directory with input data)
    Then check that the response is valid and contains expected data.
    """
    models = os.listdir("models") # ["linear_regression","logistic_regression","random_forrest"]

    for model_name in models:
        
        input_data={}

        with open("models/"+model_name+"/input.txt", "r") as file:
            input_info = json.load(file)

        for input in input_info: 
            input_data[input["name"]] = input["default"]


        # Send a POST request to the ml_predict route
        response = test_client.post(f'/{model_name}', data=input_data)

        assert response.status_code == 200
        assert b'Estimated Car Prices' in response.data
        assert b'Your Input Values' in response.data
        assert b'engine (CC)' in response.data
        assert b'Car Price Prediction' in response.data


def test_input_error(test_client):
    """
    Given a Flask application configured for testing
    When the '/<model_name>' route is posted to (for each model in models directory with input data)
    Then check that the response is valid and contains expected data.
    """
    models = os.listdir("models") # ["linear_regression","logistic_regression","random_forrest"]

    for model_name in models:
        
        input_data={}

        with open("models/"+model_name+"/input.txt", "r") as file:
            input_info = json.load(file)

        for input in input_info: 
            input_data[input["name"]] = "asdasd"


        # Send a POST request to the ml_predict route
        response = test_client.post(f'/{model_name}', data=input_data)

        assert response.status_code == 200
        assert b'Car Price Prediction' in response.data