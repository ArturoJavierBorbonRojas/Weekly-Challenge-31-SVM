#  Weekly Challenge 31: svm

##  Description
For Week 31, I bridged the gap between Machine Learning and Mathematical Optimization by programming a **Linear Support Vector Machine (SVM)** entirely from scratch using NumPy. 

Unlike standard classifiers that draw any valid boundary between data points, an SVM seeks the **optimal hyperplane**. It actively attempts to maximize the geometric margin between different classes (e.g., students at risk of dropout vs. retained students), making it one of the most robust and elegant classification algorithms in computer science.

## How it works
The core of this algorithm relies on optimizing the **Hinge Loss** function using **Gradient Descent**.
1. **Label Transformation:** SVM mathematically requires binary labels to be $-1$ and $1$ (instead of $0$ and $1$).
2. **Margin Condition:** For each data point, the model evaluates $y_i (w \cdot x_i - b) \geq 1$. 
3. **Gradient Descent Updates:**
   * If the point is correctly classified and outside the margin, the algorithm only applies a regularization penalty to the weights to prevent them from growing out of control: 
     $$w = w - \alpha (2 \lambda w)$$
   * If the point is misclassified or falls inside the margin, the algorithm aggressively adjusts both the weights and the bias to correct the boundary:
     $$w = w - \alpha (2 \lambda w - y_i x_i)$$
     $$b = b - \alpha (y_i)$$

## Technical Highlights
* **Pure Optimization Engine:** The model iteratively minimizes the Hinge Loss without relying on external solvers or black-box libraries like `scikit-learn`.
* **Hyperparameter Tuning:** The `lambda_param` controls the trade-off between achieving a smooth decision boundary and correctly classifying every single training point.

## Dependencies
* Python 3.14.3
* NumPy
* Matplotlib
