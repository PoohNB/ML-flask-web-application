


def test_home_page(test_client):
    """
    Given a Flask application configured for testing
    When the '/' route is requested (GET)
    Then check that the response is valid.
    """
    response = test_client.get('/')
    assert response.status_code == 200
    assert b'Welcome to' in response.data

