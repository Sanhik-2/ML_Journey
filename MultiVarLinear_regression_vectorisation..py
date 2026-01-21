# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 11:06:33 2026

@author: sanhi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 08:28:42 2026

@author: sanhi
"""
import numpy as np
import matplotlib.pyplot as plt
class multivariable_linearregression_numpy:
    def __init__(self,X,alpha=1e-5,e=1e-6):
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

        errors = self.linear_fn(X) - Y
        return np.sum(errors**2) / (2*self.m)
    
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
        self.dj_dw = np.dot(X.T,errors) / self.m 
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
        """
        Train the regression model using gradient descent until convergence.

        Parameters
        ----------
        X : np.array
            Input features.
        Y : np.array
            True labels.

        Returns
        -------
        None
        usage: model.train([[list of values]], [list of labels])
        """

        iter=0
        self.history=[]
        cost=float('inf')
            #self.gradient_fn(X,Y)
        while(cost>self.e):
            self.gradient_fn(X, Y)
            self.update_fn(X,Y)
            cost=self.cost_fn(X,Y)
            self.history.append(cost)
            iter+=1
            if iter % 1000 == 0:
                print(f"Iter {iter}: cost={cost:.6f}")

        print(f"Converged at iter {iter}, final cost={cost:.6f}")
        
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
    X_normalized = np.array([
        [0.000, 0.000, 0.000, 0.111, 0.091, 0.000],
        [0.057, 0.111, 0.091, 0.000, 0.000, 0.111],
        [0.143, 0.333, 0.304, 0.333, 0.182, 0.222],
        [0.229, 0.222, 0.174, 0.222, 0.091, 0.333],
        [0.286, 0.556, 0.522, 0.556, 0.545, 0.444],
        [0.429, 0.444, 0.391, 0.444, 0.318, 0.556],
        [0.571, 0.778, 0.739, 0.778, 0.682, 0.667],
        [0.714, 0.667, 0.652, 0.667, 0.545, 0.778],
        [0.857, 1.000, 1.000, 1.000, 1.000, 0.889],
        [1.000, 0.889, 0.913, 0.889, 0.909, 1.000],
    ])
    Y_normalized =np.array([0.000, 0.061, 0.152, 0.242, 0.333, 0.455, 0.606, 0.727, 0.879, 1.000])

    model = multivariable_linearregression_numpy(X_normalized,alpha=2e-5,e=5e-5)
    model.train(X_normalized, Y_normalized)

    # Test prediction
    test_sample = [0.343, 0.111, 0.091, 0.222, 0.182, 0.444]
    prediction = model.tell(np.array([test_sample]))[0]

    Y_min, Y_max = 160, 490
    y_pred_actual = prediction * (Y_max - Y_min) + Y_min
    model.plot(X_normalized, Y_normalized)

    print(f"Predicted actual price: {y_pred_actual}")
            
        
    
        
        