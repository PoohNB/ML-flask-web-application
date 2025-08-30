from flask import Flask, request, render_template, jsonify

from modules.model_utils import load_model_obj
from configs.models_conf import model_info

# Load models and scalers
model_path = "models/"
model_info = load_model_obj(model_path=model_path, model_info=model_info)


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

    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok"), 200

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host="0.0.0.0",port=80)
