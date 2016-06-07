from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, SReLU

import numpy as np

# Generate fizzbuzz input data

# Input => numbers from 100 to 1000
train_numbers = list(range(101, 500))

# Bits to encode
bits = 12

# Correct answer
def corrFizzbuzz(x):
	if x % 15 == 0:
		return "FIZZBUZZ"
	if x % 3 == 0:
		return "FIZZ"
	if x % 5 == 0:
		return "BUZZ"
	return str(x)

# Binary encode input
train_x = [[(x >> d) & 1 for d in range(bits)] for x in train_numbers]

nofizzbuzz = lambda x: [0,1][x % 3 != 0 and x % 5 != 0]
fizz = lambda x: [0,1][x % 3 == 0 and x % 5 != 0]
buzz = lambda x: [0,1][x % 3 != 0 and x % 5 == 0]
fizzbuzz = lambda x: [0,1][x % 15 == 0]

train_y = [[nofizzbuzz(x), fizz(x), buzz(x), fizzbuzz(x)] for x in train_numbers]

model = Sequential()
model.add(Dense(output_dim=100, input_dim=12))
model.add(PReLU())
model.add(Dropout(p=0.2))
model.add(Dense(output_dim=50))
model.add(PReLU())
model.add(Dropout(p=0.2))
model.add(Dense(output_dim=25))
model.add(PReLU())
model.add(Dropout(p=0.2))
model.add(Dense(output_dim=4))
model.add(Activation("softmax"))

# for a multi-class classification problem
model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_x, train_y, nb_epoch=5000, batch_size=512)

print model.summary()

# Now predict
input_numbers = list(range(1, 100))

input_x = [[(x >> d) & 1 for d in range(bits)] for x in input_numbers]
input_y = model.predict(input_x)
input_correct = [[nofizzbuzz(x), fizz(x), buzz(x), fizzbuzz(x)] for x in input_numbers]

for number, predict in zip(input_numbers, input_y):
	print(number, 
		[str(number), "Fizz", "Buzz", "FizzBuzz"][np.argmax(predict)],
		corrFizzbuzz(number))

right, wrong = 0, 0
for number, predict, correct in zip(input_numbers, input_y, input_correct):
	if np.argmax(predict) == np.argmax(correct):
		right += 1
	else:
		wrong += 1

print "Right %d Wrong %d total %d" % (right, wrong, len(input_numbers))