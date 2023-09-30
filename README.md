# ML-flask-web-application

this repository is flask website for deploy machine learning model

## quick guide

1. add your model

the model have to store in models folder

for example model random forrest

models
  - random_forrest
    - scaler.pkl
    - model.pkl
    - input.txt
   
the input.txt is the input info for the model that contain 3 thing 
1. Name of the input parameter (to be displayed on the web)
2. Input type (currently supports float and int)
3. default value for that input

input.txt
'''json
[  
{"name": "engine (CC)", "type": "int", "default": 1248},  
{"name": "max_power (bhp)", "type": "float", "default": 92.31},  
{"name": "mileage (kmpl)", "type": "float", "default": 19.37},  
{"name": "year", "type": "int", "default": 2015}  
]  
'''

2. edit nav.html

'''html
<nav>
    <ul>
        <li><a href="/">Home</a></li>
        <li class="dropdown">
            <a href="#">Car Price</a>
            <div class="dropdown-content">
                <a href="{{ url_for('ml_predict', model_name='linear_regression') }}">Linear Regression</a>
                <a href="{{ url_for('ml_predict', model_name='random_forrest') }}">Random Forrest</a>
                <a href="{{ url_for('ml_predict', model_name='logistic_regression') }}">logistic regression</a>
            </div>
        </li>
        <li class="dropdown">
            <a href="#">your manu</a>
            <div class="dropdown-content">
                <a href="{{ url_for('ml_predict', model_name='your model foler name') }}">your option name</a>
            </div>
        </li>
    </ul>
</nav>
'''

3. edit the model_info in app.py

   this may look a bit tough , but you just need to add something like
   your_model_folder_name:{"describe":"""
                                     you model describetion
                                      """,
                                "page_style":'name of css that you want to use in this end point', # you can use day,night or future ,or you can create new one
                                "prediction":you object to predict the output } # see the example in app.py
   

see tutorial how to create machine learning model! [here](https://github.com/iforgeti/ML-homework.git)

curently support 
- classification
- regression

