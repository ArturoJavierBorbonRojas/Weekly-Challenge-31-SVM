import numpy as np
import matplotlib.pyplot as plt

# Weekly Challenge 31 SVM
# Author: Ing. Arturo Javier Borbón Rojas

class SVM:
    def __init__(self, learning_rate= 0.001, lambda_param= 0.01, n_iters=1000):
        self.lr= learning_rate
        self.lambda_param= lambda_param
        self.n_iters= n_iters
        self.n_iters= n_iters
        self.w= None
        self.b= None

    def fit(self, X, y):
        n_samples, n_features= X.shape

        # SVM mathematically strictly requires labels to be -1  or 1
        y_= np.where(y<=0,-1,1)

        # Initialize weights and bias
        self.w= np.zeros(n_features)
        self.b=0

        for epoch in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Check the margin condition: y_i*(w.x-b)>=1
                condition= y_[idx]* (np.dot(x_i, self.w)-self.b) >=1

                if condition:
                    #correctly classified with enough margin: apply regularization only
                    self.w -= self.lr * (2* self.lambda_param*self.w)
                else:
                    # Misclassified or inside the margin: adjust weights and bias
                    self.w -= self.lr * (2* self.lambda_param * self.w- np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

            # Track optimization progress
            if epoch % 200== 0:
                print(f"Epoch {epoch:04d} | Wights: {self.w.round(4)} | Bias: {self.b:.4f}")

    def predict(self, X):
        """Predicts the class based on the optimized hyperplane. """
        approx= np.dot(X, self.w)-self.b
        # Convert back to standard 0 and 1 labels based on the sign
        return np.where(np.sign(approx)== -1,0,1)


# TESTING DATA SET

# Features: [Travel Distance to Campus (km), Total Absences]
X_train= np.array([
    [2.5,1], [5.0,3],[12.0,2],[8.0,4],[3.0,2], #safe students
    [25.0,12],[30.0,8],[15.0,15],[22.0,10],[18.0,14] # at risk students
])


# labels: 0 = Retained (safe), 1 = dropout (At risk)
y_train= np.array([0,0,0,0,0,1,1,1,1,1])


# Train the SVM Model
svm= SVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
svm.fit(X_train,y_train)

X_test= np.array([
    [4.0,2], # Expected 0 safe
    [20.0,11], # Expect 1 at risk
    [14.0,6] # expected borderline case

])

predictions= svm.predict(X_test)

for i, (student, pred) in enumerate(zip(X_test, predictions)):
    status=" At Risk Dropout" if pred==1 else "Safe Retained"
    print(f"Student {i+1} [Distance: {student[0]:04.1f}km, Absences: {student[1]:02.0f}]➔ {status}")


def visualize_svm(X, y, model):
    print("\n Generating visual map of the Hyperplane and Margins...")
    
    plt.figure(figsize=(10, 6))
    
    # 1. Scatter plot the historical data points
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='lime', label='Safe (Retained)', s=120, edgecolors='k', zorder=5)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='At Risk (Dropout)', s=120, edgecolors='k', zorder=5)
    
    # 2. Set up the x-axis range for the lines
    x0_min = np.amin(X[:, 0]) - 2
    x0_max = np.amax(X[:, 0]) + 2
    x0 = np.array([x0_min, x0_max])
    
    # 3. Algebraic function to solve for y (x1) in the SVM equation: w0*x0 + w1*x1 - b = offset
    def get_hyperplane_value(x, w, b, offset):
        return (offset + b - w[0] * x) / w[1]
    
    # Calculate the coordinates for the decision boundary (offset = 0)
    x1_decision = get_hyperplane_value(x0, model.w, model.b, 0)
    # Calculate the coordinates for the margins (offset = 1 and -1)
    x1_margin_pos = get_hyperplane_value(x0, model.w, model.b, 1)
    x1_margin_neg = get_hyperplane_value(x0, model.w, model.b, -1)
    
    # 4. Plot the lines
    plt.plot(x0, x1_decision, 'k-', linewidth=2.5, label='Optimal Hyperplane')
    plt.plot(x0, x1_margin_pos, 'k--', linewidth=1.5, alpha=0.7, label='Margin (+1)')
    plt.plot(x0, x1_margin_neg, 'k--', linewidth=1.5, alpha=0.7, label='Margin (-1)')
    
    # 5. Formatting the chart
    y_min = np.amin(X[:, 1]) - 2
    y_max = np.amax(X[:, 1]) + 2
    plt.ylim([y_min, y_max])
    plt.xlim([x0_min, x0_max])
    
    plt.title("SVM Optimization: Geometric Decision Boundary", fontsize=16, fontweight='bold')
    plt.xlabel("Travel Distance to Campus (km)", fontsize=12)
    plt.ylabel("Total Absences", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Fill the background to show the decision zones
    plt.fill_between(x0, x1_decision, y_max + 5, color='red', alpha=0.1)
    plt.fill_between(x0, x1_decision, y_min - 5, color='lime', alpha=0.1)
    
    plt.tight_layout()
    plt.show()

# Run the visualization function using your trained model and training data
visualize_svm(X_train, y_train, svm)