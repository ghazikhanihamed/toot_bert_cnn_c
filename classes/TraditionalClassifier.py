from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

class TraditionalClassifier:
    
    def __init__(self, X_train, y_train, X_val, y_val, X_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=10000),
            'Random Forest': RandomForestClassifier(),
            'kNN': KNeighborsClassifier(),
            'SVM': SVC(max_iter=10000),
            'MLP': MLPClassifier(max_iter=10000)
        }
        self.best_model = None
    
    def train(self):
        best_score = 0
        for name, model in self.models.items():
            param_grid = self.get_param_grid(name)
            clf = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
            clf.fit(self.X_train, self.y_train)
            score = clf.best_score_
            if score > best_score:
                best_score = score
                self.best_model = clf.best_estimator_
        return self.best_model
    
    def get_param_grid(self, name):
        if name == 'Logistic Regression':
            return {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2']
            }
        elif name == 'Random Forest':
            return {
                'n_estimators': [100, 500],
                'max_depth': [5, 10, None]
            }
        elif name == 'kNN':
            return {
                'n_neighbors': [5, 10, 15],
                'weights': ['uniform', 'distance']
            }
        elif name == 'SVM':
            return {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf']
            }
        elif name == 'MLP':
            return {
                'hidden_layer_sizes': [(10,), (50,), (100,)],
                'activation': ['relu', 'logistic']
            }
        else:
            return {}
    
    def evaluate(self, model):
        mean = 0
        std = 0
        if model is not None:
            scores = cross_val_score(model, self.X_train, self.y_train, cv=5, n_jobs=-1)
            mean = scores.mean()
            std = scores.std()
        return mean, std
    
    def test(self, model):
        y_pred = model.predict(self.X_test)
        return y_pred
    
    def save_results(self, mean, std):
        results_df = pd.DataFrame({'mean': [mean], 'std': [std]})
        results_df.to_csv('results.csv', index=False)
    
    def save_model(self, model):
        if model is not None:
            joblib.dump(model, 'best_model.joblib')
