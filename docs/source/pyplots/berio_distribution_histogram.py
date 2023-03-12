import seaborn as sns
import matplotlib.pyplot as plt
from ml.datasets import load_berio_nono

data = load_berio_nono()

data = data.replace({
    '7C_7C': '7.5 CCIR',
    '7N_7N': '7.5 NAB',
    '15C_15C': '15 CCIR',
    '15N_15N': '15 NAB'})

sns.histplot(data=data, x="label", hue="noise_type", multiple="stack")
plt.title("Distribution of the labels in the Berio-Nono dataset")
plt.xlabel("Equalization parameters")
