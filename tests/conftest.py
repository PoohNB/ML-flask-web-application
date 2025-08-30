import os
from app import create_app
import pytest
from modules.model_utils import load_model_obj,ML_predictor
from configs.models_conf import model_info

# @pytest.fixture(scope='module')
# def test_predictor():
#     model_path = "models/"
#     for conf in model_info:

#     return model_info



@pytest.fixture(scope='module')
def test_client():
    # Set the Testing configuration prior to creating the Flask application
    os.environ['CONFIG_TYPE'] = 'config.TestingConfig'
    flask_app = create_app()

    # Create a test client using the Flask application configured for testing
    with flask_app.test_client() as testing_client:
        # Establish an application context
        with flask_app.app_context():
            yield testing_client  # this is where the testing happens!