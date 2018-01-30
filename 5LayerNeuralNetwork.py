import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import  math
#Defining 5Layer Neural Network
mnist = input_data.read_data_sets("data2",one_hot=True)

batch_size = 100
learning_rate = 0.5
training_epochs = 10

with tf.name_scope("I") as scope:
   X= tf.placeholder(dtype=tf.float32,shape=[None,784],name="input")

with tf.name_scope("O") as scope:
    Y=tf.placeholder(dtype=tf.float32,shape=[None,10],name="Output")

with tf.name_scope("N") as scope:
    L=200
    M=100
    N=60
    O=30

def Weights(input_neurons,output_neurons,name):
    return tf.Variable(tf.truncated_normal(shape=[input_neurons,output_neurons]),name=name)

def Biases(outputNeurons,name):
    return tf.Variable(tf.zeros(shape=[outputNeurons]),name=name)

def scalar_summary(name,Value):
    return tf.summary_historgram(name,Value)



#Defeining Architecture
w1=Weights(input_neurons=784,output_neurons=200,name="weight1")
b1=Biases(outputNeurons=200,name="bias1")
w2=Weights(input_neurons=200,output_neurons=100,name="Weights2")
b2=Biases(outputNeurons=100,name="bias2")
w3=Weights(input_neurons=100,output_neurons=60,name="Weights3")
b3=Biases(outputNeurons=60,name="bias3")
w4=Weights(input_neurons=60,output_neurons=30,name="weight4")
b4=Biases(outputNeurons=30,name='Bias4')
w5=Weights(input_neurons=30,output_neurons=10,name="Weight5")
b5=Biases(outputNeurons=10,name="b5")

def relu_activation_layer(w1,w2,b):
    return tf.nn.relu((tf.matmul(w1,w2))+b)

layer1=relu_activation_layer(X,w1,b1)
layer2=relu_activation_layer(layer1,w2,b2)
layer3=relu_activation_layer(layer2,w3,b3)
layer4=relu_activation_layer(layer3,w4,b4)
layer5=tf.add(tf.matmul(layer4,w5),b5)
prediction=tf.nn.softmax(layer5)

def cal_loss(logits,label):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=label))

loss=cal_loss(logits=layer5,label=Y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

correct_prediction = tf.equal(tf.argmax(Y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for epoch in range(training_epochs):
        batch_count = int(mnist.train.num_examples / batch_size)
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _,l = sess.run(fetches=[optimizer,loss],feed_dict={X:batch_x,Y:batch_y})
        print("Epoch:",epoch,"\nLoss:",l)
    print("Accuracy: ", accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
    print("done")

