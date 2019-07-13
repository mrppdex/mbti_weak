from keras.models import load_model
from tensorflow import set_random_seed
import pandas as pd

model = load_model("dense_model.h5")

train_df = pd.read_csv("train_dense_dataset.csv")
eval_df = pd.read_csv("eval_dense_dataset.csv")

train_x, train_y = train_df.values[:,:-1], train_df.values[:,-1]
eval_x, eval_y = eval_df.values[:,:-1], eval_df.values[:,-1]

set_random_seed(42)
history = model.fit(train_x, train_y,
                    validation_data = (eval_x, eval_y),
                    batch_size=64,
                    epochs=25,
                    verbose=1,
                    )


print(model.evaluate(eval_x, eval_y))

model.save("trained_model.h5")
