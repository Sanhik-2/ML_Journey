# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 21:34:05 2026

@author: sanhi
"""

# -*- coding: utf-8 -*-
"""
Adaptive Alpha Linear Regression (Multi-variable)
Created Jan 2026
@author: Sanhik
"""

import matplotlib.pyplot as plt

class LinearRegAdaptive_multivar:
    def __init__(self, X, alpha=1e-3, e=1e-3, max_iters=10000,
                 alpha_min=1e-6, alpha_max=1e-1,
                 boost=1.1, backoff=0.7, patience=1000):
        """
        Initialize the regression model with adaptive alpha.

        Parameters:
            X (list of lists): Training samples.
            alpha (float): Initial learning rate.
            e (float): Convergence tolerance.
            max_iters (int): Maximum iterations.
            alpha_min (float): Minimum allowed alpha.
            alpha_max (float): Maximum allowed alpha.
            boost (float): Factor to increase alpha when improving.
            backoff (float): Factor to decrease alpha when worsening.
            patience (int): Iterations allowed without improvement.
        """
        self.alpha = alpha
        self.e = e
        self.max_iters = max_iters
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.boost = boost
        self.backoff = backoff
        self.patience = patience

        self.m = len(X[0])      # number of features
        self.w = [0.0] * self.m # weights
        self.b = 0.0            # bias
        self.n = len(X)         # samples

    def linear_fn(self, x):
        return sum(self.w[j] * x[j] for j in range(self.m)) + self.b

    def cost_fn(self, X, Y):
        return sum((Y[i] - self.linear_fn(X[i]))**2 for i in range(self.n)) / self.n

    def gradient(self, X, Y):
        total_dj_dw = [0.0] * self.m
        total_dj_db = 0.0
        for i in range(self.n):
            error = Y[i] - self.linear_fn(X[i])
            for j in range(self.m):
                total_dj_dw[j] += -2 * error * X[i][j]
            total_dj_db += -2 * error
        # average
        total_dj_dw = [g / self.n for g in total_dj_dw]
        total_dj_db /= self.n
        return total_dj_dw, total_dj_db

    def train(self, X, Y):
        prev_cost = None
        best_cost = float('inf')
        best_w, best_b = self.w[:], self.b
        since_improve = 0

        costs = []

        for it in range(1, self.max_iters+1):
            dj_dw, dj_db = self.gradient(X, Y)

            # update weights
            for j in range(self.m):
                self.w[j] -= self.alpha * dj_dw[j]
            self.b -= self.alpha * dj_db

            cost = self.cost_fn(X, Y)
            costs.append(cost)

            # adaptive alpha logic
            if cost < best_cost - self.e:
                best_cost = cost
                best_w, best_b = self.w[:], self.b
                since_improve = 0
                self.alpha = min(self.alpha * self.boost, self.alpha_max)
            else:
                since_improve += 1
                self.alpha = max(self.alpha * self.backoff, self.alpha_min)

            # early stop
            if since_improve >= self.patience:
                print(f"Early stop at {it}, best_cost={best_cost}")
                self.w, self.b = best_w, best_b
                break

            if it % 1000 == 0 or it == 1:
                print(f"Iter {it}: alpha={self.alpha:.2e}, cost={cost:.6f}")

            prev_cost = cost

        # plot cost curve
        plt.plot(range(len(costs)), costs, color='purple')
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Cost vs Iteration (Adaptive Alpha)")
        plt.show()

    def tell(self, X_test):
        return [self.linear_fn(x) for x in X_test]
    def plot_fit(self, X, Y):
        """
    Plot predicted outputs vs actual training labels.

    Parameters:
        X (list of lists): Training set.
        Y (list): True labels.

    Returns:
        None
    """
        # Get predictions
        predictions = self.tell(X)

        # Scatter plot: actual vs predicted
        plt.scatter(Y, predictions, color='blue', label='Predicted vs Actual')

        # Perfect fit line
        min_val = min(min(Y), min(predictions))
        max_val = max(max(Y), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Fit')

        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Regression Fit: Actual vs Predicted")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Training data
    X_normalized = [
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
    ]
    Y_normalized = [0.000, 0.061, 0.152, 0.242, 0.333, 0.455, 0.606, 0.727, 0.879, 1.000]

    model = LinearRegAdaptive_multivar(X_normalized, alpha=0.2, e=1e-3, max_iters=10000,
                              alpha_min=0.2, alpha_max=1e1,
                              boost=1.1, backoff=0.7, patience=20000)

    model.train(X_normalized, Y_normalized)

    # Test prediction
    test_sample = [0.343, 0.111, 0.091, 0.222, 0.182, 0.444]
    prediction = model.tell([test_sample])[0]

    Y_min, Y_max = 160, 490
    y_pred_actual = prediction * (Y_max - Y_min) + Y_min
    model.plot_fit(X_normalized, Y_normalized)

    print(f"Predicted actual price: {y_pred_actual}")