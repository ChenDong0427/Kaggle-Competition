import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Load files into DataFrames
X_train = pd.read_csv("./data/X_train.csv")
Y_train = X_train['Score']
X_submission = pd.read_csv("./data/X_test.csv")

# This is where you can do more feature selection
X_train = X_train.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'Score'])
X_submission_processed = X_submission.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'Score'])


tuned_parameters = [{'n_estimators': [200,300], 'learning_rate': [0.3,0.4], 'max_depth': [2,3]}]

# Learn the model
print("# Tuning hyper-parameters for RMSE")
print()


reg = GridSearchCV(GradientBoostingRegressor(), tuned_parameters, cv=5)

reg.fit(X_train, Y_train)

print("Best parameters set found on development set:")
print()

print(reg.best_params_)


X_submission['Score'] = reg.predict(X_submission_processed)


 #Plot a confusion matrix
#cm = confusion_matrix(Y_test, Y_test_predictions, normalize='true')
#sns.heatmap(cm, annot=True)
#plt.title('Confusion matrix of the classifier')
#plt.xlabel('Predicted')
#plt.ylabel('True')
#plt.show()

# Create the submission file
submission = X_submission[['Id', 'Score']]
submission.to_csv("./data/submission.csv", index=False)
