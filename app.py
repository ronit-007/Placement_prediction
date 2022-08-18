from flask import Flask , render_template,request
import pickle
import sklearn
import numpy as np
import warnings 
warnings.filterwarnings("ignore")



app = Flask(__name__)
model = pickle.load(open("Placement_predictio.pkl",'rb'))
salary_model = pickle.load(open("Salary_prediction.pkl",'rb'))




def PlacementPredictor(to_predict):
    data = np.array(to_predict).reshape(1,12)
    result = model.predict(data)
    return result[0]

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/",methods=["GET", "POST"])
def predict():
    ans = "_"
    
    if request.method == "POST":

        to_predict_list = [float(x) for x in request.form.values()]
        
        
        result = PlacementPredictor(to_predict_list) 
        
        if result==1:
            ans = "Placed"
            
            to_predict_list = np.append(to_predict_list,1)
            
            salary = salary_model.predict(np.array(to_predict_list).reshape(1,13))
            return render_template("index.html",prediction = ans,sal="The Expected salary is Rs.{}".format(salary[0]))
        else:
            ans = "Not Placed"
            # print(ans)
            return render_template("index.html",prediction = ans,sal="")
    


if __name__ == "__main__":
    app.run(debug = True)
    
