import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class produce:
    def weight_and_bias(size): #random produce weight and bias 
        return np.random.rand(size,1),0
                    #weight,bias

class calculate:
    def sigmoid(x): #sigmoid function
        return(1/(1+np.exp(-x)))
    
    def predict(w,b,x_test): #predict values
        total=np.dot(w.T,x_test)+b
        total=calculate.sigmoid(total)
        for i in range(total.shape[1]):
            total[0,i]=1 if total[0,i]>=0.5 else 0
        return total
    
class propagation:
    def forward_and_backward(w,b,x,y):
        #forward
        total=np.dot(w.T,x)+b
        total=calculate.sigmoid(total)
        loss=-y*np.log(total)-(1-y)*np.log(1-total)
        cost=np.sum(loss)/x.shape[1]
        
        #backward
        derivative_weight=(np.dot(x,((total-y).T)))/x.shape[1]
        temp_total=np.sum(total-y)
        derivative_bias=temp_total/x.shape[1]
        gradients={"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}
        return cost,gradients
    
    def update_w_and_b(w,b,x,y,learning_rate,number_of_iteration): #update weight and bias values
        cost_list=[]
        index_numbers=[]
        for i in range(number_of_iteration):
            cost,gradients=propagation.forward_and_backward(w,b,x,y)
            w=w-learning_rate*gradients["derivative_weight"]
            b=b-learning_rate*gradients["derivative_bias"]
            if i%(number_of_iteration/10)==0:
                cost_list.append(cost)
                index_numbers.append(i)
        plt.plot(index_numbers,cost_list)
        plt.xticks(index_numbers,rotation="vertical")
        plt.show()
        return w,b,gradients,cost_list
        
class regresion:
    def logistic_regression(x,y,x_test,y_test,learning_rate,number_of_iteration): #simplified interface
        w,b=produce.weight_and_bias(x.shape[0])
        w,b,gradients,cost_list=propagation.update_w_and_b(w,b,x,y,learning_rate,number_of_iteration)
        y_prediction=calculate.predict(w,b,x_test)
        print("Accuracy=",(100-np.mean(\
                np.abs(y_prediction-y_test))*100))

data=pd.read_csv("voice.csv")
x=data.drop(["label"],axis=1)
y=pd.DataFrame([(1 if i=="male" else 0) for i in pd.DataFrame(data["label"]).values])
x=(x-np.min(x))/(np.max(x)-np.min(x))

x,x_test,y,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

x=x.T.values
x_test=x_test.T.values
y=y.T.values
y_test=y_test.T.values

regresion.logistic_regression(x,y,x_test,y_test,1,1000)

print("\n**Done**\n")