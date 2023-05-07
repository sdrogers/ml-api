import numpy as np
import pandas as pd
from sklearn.svm import SVC
from joblib import dump

# Generate some training data
N_PER_CLASS = 50
X = np.vstack(
    (
        np.random.normal(size=(N_PER_CLASS, 2)),
        np.random.normal(size=(N_PER_CLASS, 2)) + 3
    )
)

y = np.vstack(
    (
        np.ones((N_PER_CLASS, 1), int),
        np.zeros((N_PER_CLASS, 1), int)
    )
).flatten()

X = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1]})
y = pd.DataFrame({'target': y})


svc = SVC(probability=True)
svc.fit(X, y.values.flatten())

dump(svc, 'models/svc.joblib')

# save data to json for testing training endpoint
traindata = {}
traindata['data'] = {
    'f1': list(X['x1'].values),
    'f2': list(X['x2'].values)
}
traindata['targets'] = list(y.values.flatten())

print(traindata)

