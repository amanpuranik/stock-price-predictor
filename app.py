import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from flask import Flask, render_template
from flask_bootstrap import Bootstrap
import matplotlib.pyplot as plt


data = pd.read_csv("/Users/amanpuranik/Desktop/stock 5.csv")
#print(data)

data = data[['PAST', 'FUTURE']]

predict = 'FUTURE'

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1, random_state= 10) #with the random state thing, my training dataset will not chnage
#print(x_train)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)
#print(accuracy) # this will show me the value of how accurate my line of best fit is that I just made. can determine with x% accuracy what a students grade will ne


'''with open("studentmodel.pickle", "wb") as f:
    pickle.dump(linear, f)  # the pickle feature helps save our model

pickle_in = open('studentmodel.pickle', 'rb')
linear = pickle.load(pickle_in)''' #this is just some plotting stuff


predictions = linear.predict(x_test)
for x in range(len(predictions)):
     print(predictions[x],x_test[x], y_test[x]) #i got rid of 'y_test[x]

output = predictions[x]
print(output)

x = predictions
print(x)

z = y_test
print(z)

plt.scatter(x,z, color = 'red')
plt.xlabel('Actual price')
plt.ylabel('Prediction')
#plt.show() #this part  acctually shows the graph coming up. my stock scatter plot.

y = str(x)
#print(y)



x = '2'

app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def math():
    #return(y)
    return render_template('index.html', variable = output)

if __name__ == "__main__":
    app.run(debug = True)
