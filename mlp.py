from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# loading dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# normalise pixel values to [0,1]
X_train = X_train/255.0
X_test = X_test/255.0

# onehot encode labels
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

# design model
model= Sequential([
    Flatten(input_shape=(28,28)),       # input layer; image size 28x28 pixels
    Dense(128, activation='relu'),      # hidden layers
    Dropout(0.3),                       #    # dropout 30% of neurons randomly; to prevent overfitting
    Dense(64, activation ='relu'),      #
    Dropout(0.5),                       #    # dropout 30% of neurons randomly; to prevent overfitting
    Dense(10, activation='softmax')     # output layer 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'precision'])

#model training
model.fit(X_train, y_train ,epochs=5, batch_size=32)

# evaluate model
test_loss, test_accuracy, test_precision =  model.evaluate(X_test, y_test)

#printing accuracy
print(f'\nAccuracy is {test_accuracy:.4f}')
print(f'Precision is {test_precision:.4f}')
print(f'Loss is {test_loss:.4f}')