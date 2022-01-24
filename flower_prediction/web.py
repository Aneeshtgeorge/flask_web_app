from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
    temp=[]
    sl=float(request.values['Sepal_Length'])
    temp.append(sl)
    sw=float(request.values['Sepal_Width'])
    temp.append(sw)
    pl=float(request.values['Petal_Length'])
    temp.append(pl)
    pw=float(request.values['Petal_Width'])
    temp.append(pw)
    ## temp=temp.reshape(1,-1)
    output=model.predict([temp])
    output=output.item()
    
    ## check the result.html file
    ## prediction_text is a place holder
    ## ctrl+shift+p
    return render_template ('predict.html',prediction_text="It is a {}".format(output),flower=format(output))
if __name__=='__main__':
    app.run(port=8000,debug=True)



