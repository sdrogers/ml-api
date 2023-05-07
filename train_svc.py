import numpy as np
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
        np.ones((N_PER_CLASS, 1)),
        np.zeros((N_PER_CLASS, 1))
    )
).flatten()

svc = SVC(probability=True)
svc.fit(X, y)

dump(svc, 'models/svc.joblib')

