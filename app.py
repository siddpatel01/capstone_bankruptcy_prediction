import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


app=Flask(__name__)
## Load the Random Forest Model
randomforestmodel=pickle.load(open('randomforestmodelbestmodel.pkl','rb'))
scalar=pickle.load(open('scalingreduced.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/resume')
def resume_page():
    return render_template('resume.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=randomforestmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = randomforestmodel.predict(final_input)[0]
    if output == 0:
        return render_template("home.html",prediction_text="The company has a low probability of going bankrupt")
    elif output == 1:
        return render_template("home.html", prediction_text="The company has a high probability of going bankrupt!")

if __name__=="__main__":
    app.run(debug=True)
