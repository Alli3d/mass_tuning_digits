import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = load_digits()

scaler = StandardScaler()
x_scaled = scaler.fit_transform(df.data)
pca = PCA(0.95)        # pca = PCA(n_components=10)
x_pca = pca.fit_transform(x_scaled)
# to check effect of reducing to principle components
# print(x_scaled.shape)
# print(x_pca.shape)

x_train, x_test, y_train, y_test = train_test_split(x_pca, df.target, test_size=0.2)


model_params = {
    'svm': {
        'model': svm.SVC(),
        'params': {
            'gamma': ['scale','auto'],
            'kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [1,5,10,50,100]
       }
    },
    'logistc_regression' : {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'C': [1,5,10,50,100]
        }
    },
    'gaussianNB': {
        'model': GaussianNB(),
        'params': {}
    },
    'multinomialNB': {
        'model': MultinomialNB(),
        'params': {}
    },
    'decisiontree': {
        'model': DecisionTreeClassifier(),
        'params': {}
    }
}

scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(df.data, df.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)