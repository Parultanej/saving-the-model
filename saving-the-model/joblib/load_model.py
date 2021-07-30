import joblib
# load the model
loaded_model=joblib.load('dib_79.pkl')
pred=loaded_model.predict([[10,25,38,40,0,60,15,10]])
if pred[0]==1:
    print('the person is diabatic')
else:
    print('person non diabatic')
