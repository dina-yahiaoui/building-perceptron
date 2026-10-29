import numpy as np


class Perceptron:
    # A basic binary classifier: it learns weights for the input features
    # and predicts either class 1 or class 0.
    def __init__(self, lr=0.1, epochs=100):
        # lr controls the size of each learning update; epochs is the
        # maximum number of full passes over the training data.
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        # Start with one weight per feature and a bias (intercept) of zero.
        self.w = np.zeros(X.shape[1]); self.b=0.0
        for epoch in range(self.epochs):
            errors = 0
            # Update the model once for each training example.
            for xi, yi in zip(X, y):
                error = yi - self.predict(xi)
                if error != 0:
                    # Move the weights and bias toward the correct label.
                    # A correct prediction has error 0 and makes no update.
                    self.w += self.lr * error * xi
                    self.b += self.lr * error
                    errors += 1
            # Stop early if this whole pass made no mistakes.
            if errors == 0:
                break

    def predict(self, X):
        # Compute the weighted sum for each example, then apply a threshold.
        # The result is 1 when the score is at least zero, otherwise 0.
        return np.where(X @self.w + self.b >= 0, 1, 0)

# Use a fixed random seed so the generated examples are repeatable.
rng = np.random.default_rng(42)
reussites, echecs = [], []
# Keep drawing two scores until we have the requested number of examples
# in both classes. Scores are rounded to half-point increments.
while len(reussites) < 49 or len(echecs) < 31:
    x1, x2 = np.round(rng.uniform(2, 19, 2) * 2) / 2   # notes au demi-point
    moyenne = (x1 + x2) / 2
    # Averages >= 11 are labeled as successes; averages <= 9 as failures.
    # Values between 9 and 11 are intentionally left out.
    if moyenne >= 11 and len(reussites) < 49:
        reussites.append([x1, x2])
    elif moyenne <= 9 and len(echecs) < 31:
        echecs.append([x1, x2])

# Combine the two groups into a feature matrix and matching 1/0 labels.
X = np.array(reussites + echecs)
y = np.array([1] * 49 + [0] * 31)
# Shuffle features and labels with the same indices to keep each pair aligned.
idx = rng.permutation(80)
X, y = X[idx], y[idx]

# Scale scores from roughly 0-20 to roughly 0-1 before training.
X_norm = X / 20
model = Perceptron(lr=0.1)
model.fit(X_norm, y)

# Predict for two new examples; apply the same scaling used during training.
prediction = model.predict(np.array([[14,11],[7,9.5]])/20)
print(prediction)