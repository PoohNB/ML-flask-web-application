from modules.model_utils import ML_predictor
import numpy as np


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