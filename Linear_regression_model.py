#import pandas as pd
#import numpy as np
#from matplotlib import pyplot as plt

#Linear Regression is a Machine Learning Model that we use to predict values by equating the data as a linear form. Let's start:
class linear_reg:    
    cost=0
    def linear_fn(self,w,b,x):
      f=w*x+b
      return f
  
    def cost_fn(self,w,b,x,y,m):
        cost=0
        for i in range(m):
            cost+=((y[i]-self.linear_fn(w,b,x[i]))**2)
        return cost/m


    def gradient_descent(self,fi,yi,xi,m):
        dj_dw=(2/m)*((fi-yi)*xi)
        dj_db=(2/m)*(fi-yi)
        return dj_dw,dj_db


    def update_fn(self,w,b,x,y,m):
        dj_dw=0
        dj_db=0
        w_in=0
        b_in=0
        alpha=0.0001
        for i in range(m):
            dj_dw,dj_db=self.gradient_descent(self.linear_fn(w,b,x[i]),y[i],x[i],m)
            w_in=w-(alpha*dj_dw)
            b_in=b-(alpha*dj_db)
            w=w_in
            b=b_in
        return w,b,dj_dw,dj_db

if __name__=='__main__':
        w,b,e=0,0,0.00001
        x = [1, 1.2, 1.5, 1.8, 2, 2.5, 3]   # house sizes in 1000 sq ft
        y = [150, 170, 200, 230, 250, 300, 350]          # costs (scaled, e.g. 150 = 150000)
        m=7
        dj_dw,dj_db=1,1
        model=linear_reg()
        while((abs(dj_db))>e or (abs(dj_dw))>=e):
            
            w,b,dj_dw,dj_db=model.update_fn(w,b,x,y,m)
            print(f"w={w}, b={b}, cost={model.cost_fn(w, b, x, y, m)}")
        choice=input("Wanna test this y/n")
        while(choice.upper()=='Y'):
                test=float(input("Enter size of hosue remember, 1000sq ft is written as 1."))
                print(f"The price would generally be :{model.linear_fn(w,b,test)}")
                choice=input("Wanna test this to check housing prices?y/n")
           
    