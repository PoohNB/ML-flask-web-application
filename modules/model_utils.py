
import pickle
import json

def load_model_obj(model_path:str, model_info:dict) -> dict:

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

    return model_info

import numpy as np
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