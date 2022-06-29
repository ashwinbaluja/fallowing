import data 
import tensorflow as tf 
import numpy as np 
import os 

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


#x, y, loc = data.load_data() 


x = np.load("x.np.npy").reshape((-1, 30))

y = np.load("y.np.npy").reshape((-1, 1))

print(x[0], y[0])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(30)))
model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu, use_bias=True, kernel_initializer='he_uniform'))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu, use_bias=True))

print(x.shape, y.shape)
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.MeanSquaredError()) 

model.summary()

h = model.fit(x, y, validation_split=.1, batch_size=100,epochs=100, callbacks=[cp_callback], shuffle=True)
