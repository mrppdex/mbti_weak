from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model

nn_in = Input(shape=(520,))
nn_d1 = Dense(1024, activation='relu')(nn_in)
nn_bn1 = BatchNormalization(center=True)(nn_d1)
nn_do1 = Dropout(rate=0.9)(nn_bn1)
nn_d2 = Dense(256, activation='tanh')(nn_do1)
out_ = Dense(1, activation='sigmoid')(nn_d2)

model = Model(inputs=[nn_in], outputs=[out_])
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.save("dense_model.h5")
