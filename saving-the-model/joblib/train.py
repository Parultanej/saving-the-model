import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
df=pd.read_csv(url,names=names)
print(df)
print(df.info())
x=df.iloc[:,0:8]
y=df.iloc[:,8]
print(x)
print(y)
x_train , x_test, y_train, y_test = model_selection.train_test_split(x,y , test_size = 0.20 , random_state = 101) 
 # train the model 
model = LogisticRegression() 
model.fit(x_train , y_train) 
print('[info] model has been trained') 
 #   accuracy
result = model.score(x_test , y_test) 
print(f'accuracy of the model is {result}')  
   # saving the model 
joblib.dump(model ,'dib_79.pkl')