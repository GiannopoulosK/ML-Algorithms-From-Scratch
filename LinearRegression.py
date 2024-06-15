import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, method='ols', learning_rate=0.01, epochs=1000):
        self.method = method
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coefficients = None
        
    def fit(self, X_train, y_train):
        if self.method == 'ols':  # Ordinary Least Squares
            if X_train.ndim == 1:  # Simple Linear Regression
                self._fit_simple_ols(X_train, y_train)
            else:  # Multiple Linear Regression 
                self._fit_multiple_ols(X_train, y_train)
        elif self.method == 'gradient_descent':
            self._fit_gradient_descent(X_train, y_train)
        else:
            raise ValueError("Invalid method. Choose either 'ols' or 'gradient_descent'.")
        
        
    def _fit_simple_ols(self, X_train, y_train):
        if len(X_train) != len(y_train):
            raise ValueError("Length of X_train and y_train must be the same.")
        if len(X_train) == 0:
            raise ValueError("X_train and y_train must not be empty.")
        
        # Calculate b0 and b1 with OLS
        x_mean = self._calculate_mean(X_train)  
        y_mean = self._calculate_mean(y_train)  
        
        Sxx = sum((xi - x_mean) ** 2 for xi in X_train)
        Sxy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(X_train, y_train))
        
        self.coefficients = {'b0': y_mean - (Sxy / Sxx) * x_mean, 'b1': Sxy / Sxx}
        
        
    def _fit_multiple_ols(self, X_train, y_train):
        # Add a column of ones for the intercept term
        X_with_intercept = np.column_stack([np.ones(len(X_train)), X_train])

        # Compute coefficients using the normal equation
        self.coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y_train
        
    def _fit_gradient_descent(self, X_train, y_train):
        # Implementation of gradient descent for multiple linear regression
        n_samples, n_features = X_train.shape
        self.coefficients = np.zeros(n_features + 1)  # +1 for the intercept
        
        X_with_intercept = np.column_stack([np.ones(n_samples), X_train])  # Add intercept
        
        for _ in range(self.epochs):
            gradients = np.zeros(n_features + 1)
            for i in range(n_samples):
                y_pred = X_with_intercept[i] @ self.coefficients
                error = y_pred - y_train[i]
                gradients += error * X_with_intercept[i]
            gradients *= 2 / n_samples
            self.coefficients -= self.learning_rate * gradients
        
    
    def predict(self, X_test):
        # Implementation of predict method (similar for both simple and multiple linear regression)
        if self.coefficients is None:
            raise RuntimeError("Model has not been trained yet. Please call fit() first.")
        if X_test.ndim == 1:  # Simple Linear Regression
            return self.coefficients['b0'] + self.coefficients['b1'] * X_test
        else:  # Multiple Linear Regression
            X_test_with_intercept = np.column_stack([np.ones(len(X_test)), X_test])
            return X_test_with_intercept @ self.coefficients
    
    
    @staticmethod
    def _calculate_mean(values):
        return sum(values) / len(values)
    
    @staticmethod
    def _calculate_sse(y_test, y_pred):
        if isinstance(y_test, np.ndarray) and isinstance(y_pred, np.ndarray):
            SSE = np.sum((y_test - y_pred) ** 2)
        else:
            SSE = (y_test - y_pred) ** 2
        return SSE
    
    def _calculate_ssto(self, y_test, y_pred):
        y_mean = self._calculate_mean(y_test)
        SSTO = sum((yi_test - y_mean) ** 2 for yi_test in y_test)
        return SSTO
    
    def r2_score(self, y_test, y_pred):
        SSE = self._calculate_sse(y_test, y_pred)
        SSTO = self._calculate_ssto(y_test, y_pred)
        r2_score = 1 - (SSE / SSTO)
        return r2_score
    
    def MSE(self, y_test, y_pred):
        SSE = self._calculate_sse(y_test, y_pred)
        MSE = SSE / len(y_test)    
        return MSE
    
    def RMSE(self, y_test, y_pred):
        SSE = self._calculate_sse(y_test, y_pred)
        RMSE = np.sqrt(SSE / len(y_test))    
        return RMSE
    
    def MAE(self, y_test, y_pred):
        MAE = sum(abs(yi_test - yi_pred) for yi_test, yi_pred in zip(y_test, y_pred)) / len(y_test)
        return MAE
    
    def _calculate_residuals(self, y_test, y_pred):
        if isinstance(y_test, np.ndarray) and isinstance(y_pred, np.ndarray):
            self.residuals = y_test - y_pred
        else:
            self.residuals = [yi_test - yi_pred for yi_test, yi_pred in zip(y_test, y_pred)]

    def plot_data(self, X_train, y_train):
        plt.scatter(X_train, y_train, color='blue')
        plt.xlabel('X_train')
        plt.ylabel('y_train')
        plt.title('Scatter plot of Training Data')
        plt.show()

    def plot_regression_line(self, X_train, y_train):
        plt.scatter(X_train, y_train, color='blue')
        plt.plot(X_train, self.coefficients['b0'] + self.coefficients['b1']*X_train, color='red')
        plt.xlabel('X_train')
        plt.ylabel('y_train')
        plt.title('Regression Line')
        plt.show()

    def plot_residuals(self, y_test, y_pred):
        self._calculate_residuals(y_test, y_pred)
        plt.scatter(range(len(self.residuals)), self.residuals, color='green')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.xlabel('Index')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.show()