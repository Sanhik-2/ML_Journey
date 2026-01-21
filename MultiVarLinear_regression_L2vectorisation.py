# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 14:55:53 2026

@author: sanhi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 13:44:11 2026

@author: sanhi
"""

import numpy as np
import matplotlib.pyplot as plt
class multivariable_linearregression_L2regularised:
    def __init__(self,X,alpha=1e-5,e=1e-6,lamda=0.7,patience=100000):
        """
        Initialize the multivariable linear regression model.

        Parameters
        ----------
        X : np.array
            Input feature matrix of shape (m, n), where m = samples, n = features.
        alpha : float, optional
            Learning rate for gradient descent (default = 1e-5).
        e : float, optional
            Convergence tolerance for cost function (default = 1e-6).

        Attributes
        ----------
        b : float
            Bias term.
        w : np.array
            Weight vector of shape (n,).
        m : int
            Number of training samples.
        n : int
            Number of features.
        dj_dw : np.array
            Gradient with respect to weights.
        dj_db : float
            Gradient with respect to bias.
        history : list
            Cost values recorded during training.
        """

        self.b=0
        self.m=X.shape[0]
        self.n=X.shape[1]
        self.w=np.zeros(self.n)
        self.alpha=alpha
        self.dj_dw=np.zeros(self.n)
        self.dj_db=0
        self.e=e
        self.history=[]
        self.lamda=lamda
        self.patience=patience
        
    def linear_fn(self, X):
        """
        Compute the linear function (hypothesis).

        Parameters
        ----------
        X : np.array
            Input features of shape (m, n).

        Returns
        -------
        np.array
            Predicted values of shape (m,).
        """

        return np.dot(X, self.w) + self.b
        
    def cost_fn(self,X,Y):
        """
        Compute the cost function (Mean Squared Error).

        Parameters
        ----------
        X : np.array
            Input features.
        Y : np.array
            True labels.

        Returns
        -------
        float
            Cost value.
        """
        reg_cost=np.sum(self.w**2)
        reg_cost=(self.lamda*reg_cost)/(2*self.m)
        errors = (self.linear_fn(X) - Y)
        return np.sum(errors**2) / (2*self.m)+reg_cost
    
    def gradient_fn(self,X,Y):
        """
        Compute gradients for weights and bias.

        Parameters
        ----------
        X : np.array
            Input features.
        Y : np.array
            True labels.

        Returns
        -------
        None
        """
        
        errors = self.linear_fn(X) - Y 

        grad_w = (X.T @ errors) / self.m + (self.lamda / self.m) * self.w
        
        self.dj_dw = grad_w
        self.dj_db = np.sum(errors) / self.m    
        
    def update_fn(self,X,Y):
        """
        Update weights and bias using computed gradients.

        Parameters
        ----------
        X : np.array
            Input features.
        Y : np.array
            True labels.

        Returns
        -------
        None
        """

        self.w=self.w - self.alpha*(self.dj_dw)
        self.b=self.b-self.alpha*self.dj_db
        
    def train(self, X, Y):
        best_cost = float('inf')
        patience_counter = 0
        it = 0

        while True:
            self.gradient_fn(X, Y)
            self.update_fn(X, Y)
            cost = self.cost_fn(X, Y)
            self.history.append(cost)

            if cost < best_cost - self.e:
                best_cost = cost
                patience_counter = 0
            else:
                    patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at iter {it}, cost={cost:.6f}")
                break
        
            # Optional gradient-norm stop
            if np.linalg.norm(self.dj_dw) < 1e-6 and abs(self.dj_db) < 1e-6:
                print(f"Stopped at iter {it}, gradients vanished, cost={cost:.6f}")
                break

            if it % 1000 == 0:
                print(f"Iter {it}: cost={cost:.6f}")

            it += 1
        
    def tell(self,X_train):
        """
        Predict output for given input features.

        Parameters
        ----------
        X_train : np.array
            Input features.

        Returns
        -------
        np.array
            Predicted values.
        usage: model.tell([[list of values]])
        """

        return(self.linear_fn(X_train))
    
    def plot(self, X, Y):
        """
        Plot actual vs predicted values and cost convergence after training.

        Parameters
        ----------
        X : np.array
            Input features.
        Y : np.array
            True labels.

        Returns
        -------
        None
        usage: model.plot(X_train, Y_train)
        """

   

        # Predictions
        preds = self.tell(X)

        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Actual vs Predicted
        axs[0].scatter(Y, preds, color='blue', label='Predicted vs Actual')
        axs[0].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', label='Perfect Fit')
        axs[0].set_xlabel("Actual Values")
        axs[0].set_ylabel("Predicted Values")
        axs[0].set_title("Actual vs Predicted")
        axs[0].legend()
        axs[0].grid(True)

        # Cost Convergence
        if self.history:
            axs[1].plot(range(len(self.history)), self.history, color='green')
            axs[1].set_xlabel("Iteration")
            axs[1].set_ylabel("Cost")
            axs[1].set_title("Cost Convergence")
            axs[1].grid(True)

        plt.tight_layout()
        plt.show()
    
if __name__ == "__main__":
    # Training data
    X = np.array([
    [0.85,  2, 1, 20, 15, 7],
    [1.2, 3, 2, 10, 10, 5],
    [1.5, 3, 2, 5,  8,  4],
    [1.8, 4, 2, 2,  5,  3],
    [2., 4, 3, 1,  3,  2],
    [2.2, 5, 3, 15, 12, 6],
    [2.5, 5, 4, 8,  7,  3],
    [2.7, 5, 4, 3,  4,  2],
    [3, 6, 5, 1,  2,  1],
    [3.2, 6, 5, 25, 20, 8],
    [1.0, 2, 1, 30, 20, 8],   # small, old, high crime, far away
    [2.8, 5, 4, 4, 5, 2],     # upscale, young, low crime
    [3.5, 6, 5, 10, 8, 3],    # very large, moderate age, moderate crime
    [1.3, 3, 2, 18, 14, 6],   # compact, older, higher crime
    [2.0, 4, 3, 12, 9, 4]
    ])

    Y = np.array([
    12,
    18,
    22,
    28,
    32,
    30,
    35,
    40,
    45,
    38,
    15,
    42,
    48,
    22,
    34
    ])


    model = multivariable_linearregression_L2regularised(X,alpha=5e-4,e=5e-2,lamda=0.4,patience=10000000)
    model.train(X, Y)

    # Test prediction
    X_test = np.array([
    [1.400, 3, 2, 12, 9, 5],    # mid-size, moderate age, medium distance
    [2.100, 4, 3, 6, 6, 3],     # larger, newer, closer to city
    [2.800, 5, 4, 4, 5, 2],     # upscale, young, low crime
    [3.300, 6, 5, 20, 18, 7],   # big but older, far from city
    [1.000, 2, 1, 30, 20, 8]    # small, old, high crime, far away
        ])
    #Y_test = np.array([
   # 200000,
    #310000,
    #390000,
   # 360000,
    #150000
    #])
    prediction=model.tell(X_test)

   
    model.plot(X,Y)
    for i in range(5):
        print(f"Predicted actual price: {prediction[i]*10000}")
    