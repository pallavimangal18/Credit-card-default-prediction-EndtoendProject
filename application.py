from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data = CustomData (
            LIMIT_BAL = float(request.form['LIMIT_BAL']),
            SEX = request.form['SEX'],
            EDUCATION = request.form['EDUCATION'],
            MARRIAGE = request.form['MARRIAGE'],
            AGE = float(request.form['AGE']),
            PAY_0 = float(request.form['PAY_0']),
            PAY_2 = float(request.form['PAY_2']),
            PAY_3 = float(request.form['PAY_3']),
            PAY_4 = float(request.form['PAY_4']),
            PAY_5 = float(request.form['PAY_5']),
            PAY_6 = float(request.form['PAY_6']),
            BILL_AMT1 = float(request.form['BILL_AMT1']),
            BILL_AMT2 = float(request.form['BILL_AMT2']),
            BILL_AMT3=float(request.form['BILL_AMT3']),
            BILL_AMT4 = float(request.form['BILL_AMT4']),
            BILL_AMT5 = float(request.form['BILL_AMT5']),
            BILL_AMT6 = float(request.form['BILL_AMT6']),
            PAY_AMT1 = float(request.form['PAY_AMT1']),
            PAY_AMT2 = float(request.form['PAY_AMT2']),
            PAY_AMT3 = float(request.form['PAY_AMT3']),
            PAY_AMT4 = float(request.form['PAY_AMT4']),
            PAY_AMT5 = float(request.form['PAY_AMT5']),
            PAY_AMT6 = float(request.form['PAY_AMT6']),
        )
              
        

        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)
    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)    