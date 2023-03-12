from ml.datasets import load_pretto
from ml.visualization import plot_confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

SEED = 42

X, y = load_pretto(return_X_y=True)
X = X.drop(columns=['noise_type'], axis=1)
labels = y.unique()

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=SEED)

rfc = RandomForestClassifier(random_state=SEED,
                             n_jobs=-1,
                             max_features='log2',
                             n_estimators=111,
                             min_samples_leaf=1,
                             criterion='log_loss')

rfc.fit(X_train, y_train)

fig, axes = plt.subplots(figsize=(9, 9), constrained_layout=True)

y_pred = rfc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
plot_confusion_matrix(y_test,
                        y_pred,
                        ax=axes,
                        title=f"Random Forest - accuracy {acc:.3f}",
                        labels=labels)
