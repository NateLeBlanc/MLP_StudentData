#Tri-class Classification (L,M,H) of students based on academics
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

tf.enable_eager_execution()

#import data set
stu_data = pd.read_csv("../stu_aca/stuDataSet.csv")
#Give names of columns based on order in csv
column_names = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion', 'Class']
print(stu_data.shape)
#seperate column names and classifcation label
feature_names = column_names[:-1]
label_name = column_names[-1]
#classification names
class_names = ['L', 'M', 'H']

#Give number of examples
batch_size = 16
###Makes csv into trainable dataset using tensorflow###
train_dataset = tf.contrib.data.make_csv_dataset(
	"../stu_aca/stuDataSet_train.csv",
	batch_size,
	column_names=column_names,
	label_name=label_name,
	num_epochs=1)
#With eager execution to build features and labels of dataset
features, labels = next(iter(train_dataset))
#print(features)

#Build small scatter plot from dataset
"""plt.scatter(features['raisedhands'], features['VisITedResources'], label=class_names)
plt.xlabel("Raised Hands")
plt.ylabel("VisitedResource")
plt.show()"""

###Method to pack features into a single array###
def pack_features_vector(features, labels):
	features = tf.stack(list(features.values()), axis=1)
	return features, labels
train_dataset = train_dataset.map(pack_features_vector)
features, train = next(iter(train_dataset))
features = tf.cast(features, tf.float32)
#print(features[:5])

###Create model using keras(input, hidden, output)###
model = tf.keras.Sequential([
	tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
	tf.keras.layers.Dense(10, activation=tf.nn.relu),
	tf.keras.layers.Dense(3)
])
#Prediction using model
predictions = model(features)
#print(predictions[:5])
#print("Predictions: {}".format(tf.argmax(predictions, axis=1)))
#print("     Lables: {}".format(labels))

###Define loss function###
def loss(model, x, y):
	y_ = model(x)
	return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
#Calculate loss of model
l = loss(model, features, labels)
print("Loss test: {}".format(l))

###Define gradient function###
def grad(model, inputs, targets):
	with tf.GradientTape() as tape:
		loss_value=loss(model, inputs, targets)
	return loss_value, tape.gradient(loss_value, model.trainable_variables)
#Choose best gradient descent model
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
global_step = tf.Variable(0)
#Calculate single Opt step
loss_value, grads = grad(model, features, labels)
print("Step {}, Initial Loss: {}".format(global_step.numpy(), loss_value.numpy()))
optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)
print("Step: {},        Loss: {}".format(global_step.numpy(), loss(model, features, labels).numpy()))
#Training Loop
from tensorflow import contrib
tfe = contrib.eager
#Keep results for graphing
train_loss_results = []
train_accuracy_results = []

###Run Through traing data num_epoch times###
num_epochs=201
for epoch in range(num_epochs):
	epoch_loss_avg = tfe.metrics.Mean()
	epoch_accuracy = tfe.metrics.Accuracy()
	#Using batches of 32
	for x, y in train_dataset:
		x = tf.cast(x, tf.float32)
		#Optimize
		loss_value, grads = grad(model, x, y)
		optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)
		#Track
		epoch_loss_avg(loss_value)
		#Comare prediction to actual
		epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)
	#End epoch
	train_loss_results.append(epoch_loss_avg.result())
	train_accuracy_results.append(epoch_accuracy.result())
	#Print every 50 runs
	if epoch % 50 == 0:
		print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))

#Visualize loss function
"""fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')
axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()"""

#Evaluate
test_data = tf.contrib.data.make_csv_dataset(
	"../stu_aca/stuDataSet_test.csv",
	batch_size,
	column_names=column_names,
	label_name='Class',
	num_epochs = 1,
	shuffle=False)
test_data = test_data.map(pack_features_vector)

test_accuracy =tfe.metrics.Accuracy()
for(x, y) in test_data:
	x = tf.cast(x, tf.float32)
	logits = model(x)
	predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
	test_accuracy(predictions, y)
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
