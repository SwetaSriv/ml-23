
from sklearn.linear_model import LinearRegression
from joblib import dump

# Train a simple linear regression model (replace this with your actual training code)
X_train = [[1], [2], [3]]
y_train = [2, 4, 6]
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a file (e.g., model.joblib)
dump(model, 'model.joblib')
