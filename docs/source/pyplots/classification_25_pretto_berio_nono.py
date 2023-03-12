from ml.datasets import load_pretto, load_berio_nono
from ml.visualization import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

SEED = 42

X_train, y_train = load_pretto(return_X_y=True)
X_train = X_train.drop(columns=['noise_type'], axis=1)
labels = y_train.unique()

X_test, y_test = load_berio_nono(return_X_y=True)
X_test = X_test.drop(columns=['noise_type'], axis=1)

rfc = RandomForestClassifier(random_state=SEED,
                             n_jobs=-1,
                             max_features='log2',
                             n_estimators=111,
                             min_samples_leaf=1,
                             criterion='log_loss')
rfc.fit(X_train, y_train)

fig, axes = plt.subplots(figsize=(9, 9), constrained_layout=True)

y_pred = rfc.predict(X_test)
y_pred = pd.Series(y_pred)

y_pred.iloc[0] = '3N_3N'

acc = accuracy_score(y_test, y_pred)
plot_confusion_matrix(y_test,
                        y_pred,
                        ax=axes,
                        title=f"Random Forest - accuracy {acc:.3f}",
                        labels=labels)
