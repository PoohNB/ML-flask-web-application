import os 
import sys
import random
# Get the current directory (where this script is located)
current_dir = os.path.dirname(__file__)

# Get the parent directory (project root) and add it to the Python path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


import pytest
from app import create_app
from pathlib import Path



@pytest.fixture()
def app():
    app = create_app()
    app.config.update({
        "TESTING": True,
    })

    # other setup can go here

    yield app

    # clean up / reset resources here


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def runner(app):
    return app.test_cli_runner()

# def test_request_example(client):
#     response = client.get("/")
#     assert b"<h1>Welcome to PooH's machine learning</h1>" in response.data

# get the resources folder in the tests folder
# resources = Path(__file__).parent / "resources"
# "picture": (resources / "picture.png").open("rb"),

import json

def test_ml_predict(client):
    # Create a test client

    # Define a model name for testing
    models = os.listdir("models") # ["linear_regression","logistic_regression","random_forrest"]

    # Create a sample input data for testing


    for model_name in models:
        
        input_data={}

        with open("models/"+model_name+"/input.txt", "r") as file:
            input_info = json.load(file)

        for input in input_info: 
            input_data[input["name"]] = input["default"]


        # Send a POST request to the ml_predict route
        response = client.post(f'/{model_name}', data=input_data)

        # Check if the response is valid
        assert response.status_code == 200

        # Check if the response contains the expected content (You can customize this based on your response format)
        response_data = json.loads(response.data)
        
        
        # Assuming your response contains 'car_price'
        assert 'car_price' in response_data

        # Assuming your response contains 'input_values'
        assert 'input_values' in response_data

        # Assuming your response contains 'model_input'
        assert 'model_input' in response_data

        # Assuming your response contains 'sender'
        assert 'sender' in response_data

        # check input
        assert len(input_info) == len(response_data['input_values'])

        # Check if the output is as expected
        if input_data["year"] < 1886:
            assert isinstance(response_data['car_price'], str)
        else:
            assert isinstance(response_data['car_price'], (int, float))