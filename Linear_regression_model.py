#import pandas as pd
#import numpy as np


#Linear Regression is a Machine Learning Model that we use to predict values by equating the data as a linear form. Let's start:
import matplotlib.pyplot as plt
class Linear_Reg_Single_Var:
    def __init__(self, alpha=0.0001,e=1e-5,max_parameters=10000,w=1,b=1):
        self.cost=0
        self.alpha=alpha
        self.e=e
        self.max_parameters=max_parameters
        self.w=w
        self.b=b
        
    def linear_fn(self,x):
       """ Usage:
            Parameters:w,b: integers representing w,b at function f: w*x+b(defined earlier)
                        x: lists representing training set(x)
            Returns : f an integer that contains the output of w*x+b
        """
        
       f=self.w*x+self.b
       return f
  
    def cost_fn(self,x,y,m):
        """
        Usage:
            Parameters:w,b: integers representing w,b at function f: w*x+b(defined earlier)
                        x,y: lists representing training set(x) and labels(y)
                        m:length of array
            Returns : cost
        """
        cost=0
        for i in range(m):
            cost+=((y[i]-self.linear_fn(x[i]))**2)
        return cost/m


    def gradient_descent(self,fi,yi,xi,m):
        """
        Usage:
            Parameters: fi,yi,xi=integers, representing linear function at i, x at , y at i
                        m:length of array
            Returns : tuple containing dj_dw, dj_db(gradient at point.)
        """
        dj_dw=(2/m)*((fi-yi)*xi)
        dj_db=(2/m)*(fi-yi)
        return dj_dw,dj_db


    def update_fn(self,x,y,m):
        """
        Usage:
            Parameters: w,b: integers
                        x,y:list of input and output values
                        m:length of array
            returns tuple containing w,b,dj_dw,dj_db
        """
        dj_dw=0
        dj_db=0
        w_in=0
        b_in=0
        
        for i in range(m):
            dj_dw,dj_db=self.gradient_descent(self.linear_fn(x[i]),y[i],x[i],m)
            w_in=self.w-(self.alpha*dj_dw)
            b_in=self.b-(self.alpha*dj_db)
            self.w=w_in
            self.b=b_in
        return self.w,self.b,dj_dw,dj_db

    def train(self,X,Y):
        """
        Parameters: X: list of array containing training values
                    Y: list of values containing output values
        Returns: None
        usage: model.train([list of values])
        
        """
        m=len(X)
        dj_dw, dj_db = 1, 1
        iters = 0
        while (abs(dj_db) > self.e or abs(dj_dw) >= self.e) and iters < self.max_parameters:
           self.update_fn(X, Y, m)
           iters += 1
           print(f"Iter {iters}: w={self.w}, b={self.b}, cost={self.cost_fn( X, Y, m)}")
    def tell(self, X_test):
        """
        Parameters:
            X_test (list or array): Input features to predict.
            Returns:
                list: Predicted values.
                Usage:
                    model.tell([2.2])
                    """

        return [self.linear_fn(x) for x in X_test]
    def plot(self,X,Y):
        """
        Parameters:
            X: Input features to predict.
            Y:Labels
            Returns:
                None.
                Plots the data to showcase visually.
                Usage:
                    model.plot(X,Y)
        """
        plt.scatter(X, Y, color='blue', label='Training Data')
        plt.plot(X, [self.w*x + self.b for x in X], color='red', label='Regression Line')
        plt.xlabel("House size (1000 sq ft)")
        plt.ylabel("Price (scaled)")
        plt.legend()
        plt.show()
    
   
if __name__ == "__main__":
    # Training data
    X = [1, 1.2, 1.5, 1.8, 2, 2.5, 3]   # house sizes in 1000 sq ft
    Y = [150, 170, 200, 230, 250, 300, 350]  # prices (scaled)

    # Create model
    model = Linear_Reg_Single_Var(alpha=0.0001, e=1e-5, max_parameters=10000)

    # Train model
    model.train(X, Y)

    # Test prediction
    test_value = 2.2
    prediction = model.tell([test_value])
    print(f"\nPredicted price for {test_value} (2200 sq ft): {prediction[0]}")
    model.plot(X,Y)