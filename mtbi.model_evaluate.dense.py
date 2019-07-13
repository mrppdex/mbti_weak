from keras.models import load_model
from tensorflow import set_random_seed
import pandas as pd

from sklearn import metrics

model = load_model("trained_model.h5")

eval_df = pd.read_csv("eval_dense_dataset.csv")

eval_x, eval_y = eval_df.values[:,:-1], eval_df.values[:,-1]

y_probs = model.predict(eval_x)
y_hat = (y_probs > 0.5).astype(int)
y_val = eval_y

print("f1-score: ", metrics.f1_score(y_val, y_hat))
print("acc-score: ", metrics.accuracy_score(y_val, y_hat))

fpr, tpr, thresholds = metrics.roc_curve(y_val, y_probs)
print("auc-score: ", metrics.auc(fpr, tpr))

print("balanced-accuracy-score: ", metrics.balanced_accuracy_score(y_val, y_hat))
print("\nConfusion Matrix:")
print(metrics.confusion_matrix(y_val, y_hat))
