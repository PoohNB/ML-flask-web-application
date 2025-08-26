from flask import Flask, request, render_template, send_from_directory
import pickle
import numpy as np
import json
from sklearn.preprocessing import PolynomialFeatures


class ML_predictor:

    def __init__(self,post_process=None,bias=False,catagory_label=range(20),model_type="regression",polynomial_degree=0):

        self.polynomial_degree = polynomial_degree
        self.type = model_type
        self.catagory_label = catagory_label
        self.bias = bias
        self.post_process=post_process

    def predict(self,model,input_values):


        input_array = np.array(input_values).reshape(1, -1)

        scaled_input = model["scaler"].transform(input_array)


        if self.polynomial_degree and self.bias:
            poly = PolynomialFeatures(degree = self.polynomial_degree, include_bias=True)
            scaled_input = poly.fit_transform(scaled_input)

        elif self.polynomial_degree:

            poly = PolynomialFeatures(degree = self.polynomial_degree, include_bias=False)
            scaled_input = poly.fit_transform(scaled_input)

        elif self.bias :
            scaled_input = np.concatenate(([[1]], scaled_input), axis=1)


        if self.type == "regression":

            car_price = model["model"].predict(scaled_input)[0]

            if self.post_process:
                car_price = self.post_process(car_price)

            if car_price == float('inf') or car_price == float('-inf'):
                output = "It's priceless, you never have enough money for it "
            else:
                output = round(car_price)

        elif self.type == "classification":

            car_price = model["model"].predict(scaled_input)[0]
            output = self.catagory_label[car_price]

        else:

            output = "sorry,I set something wrong"


        return output



# Model input details
model_info =   {


                "linear_regression":{"describe":"""
                                This algorithm is linear regression used polynomial tranfrom tachnique,
                                 Input the information about car inside the block below and click 'predict'. it will show car price result. 
                                Eventhought it's score no better than random forrest one,
                                      this model is the best linear regression model from more than hundred experiments on parameter. 
                                      """,
                                "page_style":'night',
                                "prediction":ML_predictor(post_process=np.exp,model_type="regression",polynomial_degree = 2,bias= True)},

                "random_forrest":{"describe":""" 
                                This algorithm is Random Forest.This platform allows you to input relevant information about a car and receive a predicted price based on our machine learning model. 
                                 Simply fill in the block below and click 'Predict' to see the estimated price. 
                                  If you're unsure about any information, don't worry; it will be filled with default data.""",
                                 "page_style":'day',
                                 "prediction":ML_predictor(post_process=np.exp,model_type="regression")},
                                 
                "logistic_regression":{"describe":"""
                                This algorithm is logistic regression , It predict 4 level of price.  
                                       Input the information about car inside the block below and click 'predict' to see if this car price are "Budget-Friendly","Mid-Range","Premium" or "Luxury".                                         
                                        """,
                                        "page_style":'future',
                                        "prediction":ML_predictor(model_type="classification",
                                                               catagory_label={0:"Budget-Friendly",1:"Mid-Range",2:"Premium",3:"Luxury"},
                                                               bias=True)} 
                }


# Load models and scalers
model_path = "models/"
for k,v in model_info.items():

    with open(model_path+k+'/scaler.pkl', 'rb') as file:
        model_info[k]["scaler"] = pickle.load(file)

    try:

        with open(model_path+k+'/model.model', 'rb') as file:
            model_info[k]["model"] = pickle.load(file)
    except :

        with open(model_path+k+'/model.pkl', 'rb') as file:
            model_info[k]["model"] = pickle.load(file)



    with open(model_path+k+"/input.txt", "r") as file:
        model_info[k]["input"] = json.load(file)


def create_app():

    app = Flask(__name__)


    @app.route('/')
    def main_home():
        return render_template('home.html')

    @app.route('/<model_name>', methods=['GET', 'POST'])
    def ml_predict(model_name):
        error_message = None

        if request.method == 'POST':
            input_values = []

            for input_info in model_info[model_name]["input"]:
                input_name = input_info["name"]
                input_type = input_info["type"]
                user_input = request.form.get(input_name)

                if user_input == '':
                    user_input = input_info["default"]
                try:
                    # Convert input to the specified type
                    if input_type == "float":
                        user_input = float(user_input)
                    elif input_type == "int":
                        user_input = int(user_input)

                    input_values.append(user_input) 

                    if input_name =="year" and user_input < 1886:
                        return render_template('result.html', 
                                            car_price='Did you know? first car in the world born in 1886 known as the "Motorwagen"', 
                                            input_values=input_values,
                                                model_input=model_info[model_name]["input"],
                                                sender=model_name)
        
                except:
                    error_message = f"Invalid input for {input_name}. Please enter a valid number."
                    break
            
            if error_message:
                return render_template('index.html', 
                                    model_input=model_info[model_name]["input"] ,
                                    describe= model_info[model_name]["describe"], 
                                    page_style = model_info[model_name]["page_style"], 
                                    error_message=error_message)
            


            # Predict car price using both 
            car_price = model_info[model_name]["prediction"].predict(model_info[model_name],input_values)
                
            return render_template('result.html', 
                                car_price=car_price, 
                                input_values=input_values,
                                    model_input=model_info[model_name]["input"],
                                    sender=model_name)

        return render_template('index.html', 
                            model_input=model_info[model_name]["input"] ,
                            describe= model_info[model_name]["describe"], 
                            page_style = model_info[model_name]["page_style"])

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host="0.0.0.0",port=80)
