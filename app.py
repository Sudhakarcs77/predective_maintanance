from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/submit', methods=['POST'])
# def submit():
#     name = request.form.get('name')
#     email = request.form.get('email')

#     # Process the form data (you can save it to a database, etc.)
#     # For now, let's just print the values
#     print(f'Name: {name}, Email: {email}')

#     return 'Form submitted!'

@app.route('/predict', methods=['GET','POST'])
def predict():
    temperature = request.form.get('temperature')
    speed = request.form.get('speed')
    tool_wear = request.form.get('tool_wear')

    # Perform any processing you need with the form data
    in_ = [temperature,speed,tool_wear]
    model = pickle.load(open("model_lasso.pickle","rb"))
    scaler = pickle.load(open("scaler.pickle","rb"))
    xin = scaler.transform([in_])
    print("x: ",in_)
    yout=model.predict(xin)
    print("y: ",yout)

    return render_template('predict.html', temperature=temperature, speed=speed, tool_wear=tool_wear,yout=yout)

@app.route('/eda', methods=['POST'])
def eda():
    # Perform any processing you need before rendering the page
    return render_template('eda.html')


if __name__ == '__main__':
    app.run(debug=True)
