from ml.datasets import load_pretto, load_berio_nono
from ml.visualization import plot_confusion_matrix
from ml.classification import evaluate_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd

data1 = load_pretto()
data2 = load_berio_nono()

data = pd.concat([data1, data2])
X = data.drop(columns=['noise_type', 'label'], axis=1)
y = data.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

SEED = 42

rfc = evaluate_model(RandomForestClassifier(random_state=SEED,
                                            max_features='log2',
                                            criterion='log_loss',
                                            min_samples_leaf=1),
                     X_train,
                     y_train,
                     params={
                         'n_estimators': [x for x in range(90, 121, 10)],
                     })

rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(y_test,
                      y_pred,
                      labels=y_train.unique(),
                      title=f'All, {accuracy_score(y_test, y_pred):.3f}',
                      ax=ax)
