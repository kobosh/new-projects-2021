# Import Required Libraries

from flask import Flask, render_template, request
import pickle


# Initialise the object to run the flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the pickled model file
model = pickle.load(open('model.pkl', 'rb+'))


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        print(features)
        labels = model.predict([features])
        print(labels)

        species = labels[0]
        # If species is 0 = setosa, if species is 1 = VersiColor, if species is 2 = Virginica
        if species == 0:
            result = "Iris-Setosa"
        elif species == 1:
            result = "Iris-VersiColor"
        else:
            result = "Iris-Virginica"
        return render_template('index.html', result=result)


# It is the starting point of code
if __name__ == '__main__':
    # We need to run the app to run the server this will change when deployed on AWS
    app.run(debug=False)
