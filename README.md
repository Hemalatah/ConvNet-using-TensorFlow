# ConvNet-using-TensorFlow
Build and train a ConvNet in TensorFlow for a classification problem

1.0 - TensorFlow model
In the previous assignment, you built helper functions using numpy to understand the mechanics behind convolutional neural networks. Most practical applications of deep learning today are built using programming frameworks, which have many built-in functions you can simply call.

As usual, we will start by loading in the packages.

(refer tens.py)

Run the next cell to load the "SIGNS" dataset you are going to use.

(refer tens.py)

As a reminder, the SIGNS dataset is a collection of 6 signs representing numbers from 0 to 5.

(refer images)

The next cell will show you an example of a labelled image in the dataset. Feel free to change the value of index below and re-run to see different examples.

(refer tens.py)

In Course 2, you had built a fully-connected network for this dataset. But since this is an image dataset, it is more natural to apply a ConvNet to it.

To get started, let's examine the shapes of your data.

(refer tens.py)

1.1 - Create placeholders
TensorFlow requires that you create placeholders for the input data that will be fed into the model when running the session.

Exercise: Implement the function below to create placeholders for the input image X and the output Y. You should not define the number of training examples for the moment. To do so, you could use "None" as the batch size, it will give you the flexibility to choose it later. Hence X should be of dimension [None, n_H0, n_W0, n_C0] and Y should be of dimension [None, n_y].

(refer tens.py)

Expected Output

X = Tensor("Placeholder:0", shape=(?, 64, 64, 3), dtype=float32)
Y = Tensor("Placeholder_1:0", shape=(?, 6), dtype=float32)


1.2 - Initialize parameters
You will initialize weights/filters  W1W1  and  W2W2  using tf.contrib.layers.xavier_initializer(seed = 0). You don't need to worry about bias variables as you will soon see that TensorFlow functions take care of the bias. Note also that you will only initialize the weights/filters for the conv2d functions. TensorFlow initializes the layers for the fully connected part automatically. We will talk more about that later in this assignment.

Exercise: Implement initialize_parameters(). The dimensions for each group of filters are provided below. Reminder - to initialize a parameter  WW  of shape [1,2,3,4] in Tensorflow, use:

W = tf.get_variable("W", [1,2,3,4], initializer = ...)

(refer tens.py)

Expected Output:

W1 =	[ 0.00131723 0.14176141 -0.04434952 0.09197326 0.14984085 -0.03514394 
-0.06847463 0.05245192]
W2 =	[-0.08566415 0.17750949 0.11974221 0.16773748 -0.0830943 -0.08058 
-0.00577033 -0.14643836 0.24162132 -0.05857408 -0.19055021 0.1345228 
-0.22779644 -0.1601823 -0.16117483 -0.10286498]


1.2 - Forward propagation
In TensorFlow, there are built-in functions that carry out the convolution steps for you.

tf.nn.conv2d(X,W1, strides = [1,s,s,1], padding = 'SAME'): given an input  XX  and a group of filters  W1W1 , this function convolves  W1W1 's filters on X. The third input ([1,f,f,1]) represents the strides for each dimension of the input (m, n_H_prev, n_W_prev, n_C_prev). You can read the full documentation here

tf.nn.max_pool(A, ksize = [1,f,f,1], strides = [1,s,s,1], padding = 'SAME'): given an input A, this function uses a window of size (f, f) and strides of size (s, s) to carry out max pooling over each window. You can read the full documentation here

tf.nn.relu(Z1): computes the elementwise ReLU of Z1 (which can be any shape). You can read the full documentation here.

tf.contrib.layers.flatten(P): given an input P, this function flattens each example into a 1D vector it while maintaining the batch-size. It returns a flattened tensor with shape [batch_size, k]. You can read the full documentation here.

tf.contrib.layers.fully_connected(F, num_outputs): given a the flattened input F, it returns the output computed using a fully connected layer. You can read the full documentation here.

In the last function above (tf.contrib.layers.fully_connected), the fully connected layer automatically initializes weights in the graph and keeps on training them as you train the model. Hence, you did not need to initialize those weights when initializing the parameters.

Exercise:

Implement the forward_propagation function below to build the following model: CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED. You should use the functions above.

In detail, we will use the following parameters for all the steps:

 - Conv2D: stride 1, padding is "SAME"
 - ReLU
 - Max pool: Use an 8 by 8 filter size and an 8 by 8 stride, padding is "SAME"
 - Conv2D: stride 1, padding is "SAME"
 - ReLU
 - Max pool: Use a 4 by 4 filter size and a 4 by 4 stride, padding is "SAME"
 - Flatten the previous output.
 - FULLYCONNECTED (FC) layer: Apply a fully connected layer without an non-linear activation function. Do not call the softmax here. This will result in 6 neurons in the output layer, which then get passed later to a softmax. In TensorFlow, the softmax and cost function are lumped together into a single function, which you'll call in a different function when computing the cost. 

(refer tens.py)

Expected Output:

Z3 =	[[-0.44670227 -1.57208765 -1.53049231 -2.31013036 -1.29104376 0.46852064] 
[-0.17601591 -1.57972014 -1.4737016 -2.61672091 -1.00810647 0.5747785 ]]


1.3 - Compute cost
Implement the compute cost function below. You might find these two functions helpful:

tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y): computes the softmax entropy loss. This function both computes the softmax activation function as well as the resulting loss. You can check the full documentation here.
tf.reduce_mean: computes the mean of elements across dimensions of a tensor. Use this to sum the losses over all the examples to get the overall cost.
Exercise: Compute the cost below using the function above.

(refer tens.py)

Expected Output:

cost =	2.91034


1.4 Model
Finally you will merge the helper functions you implemented above to build a model. You will train it on the SIGNS dataset.

You have implemented random_mini_batches() in the Optimization programming assignment of course 2. Remember that this function returns a list of mini-batches.

Exercise: Complete the function below.

The model below should:

create placeholders
initialize parameters
forward propagate
compute the cost
create an optimizer
Finally you will create a session and run a for loop for num_epochs, get the mini-batches, and then for each mini-batch you will optimize the function. Hint for initializing the variables

(refer tens.py)

Run the following cell to train your model for 100 epochs. Check if your cost after epoch 0 and 5 matches our output. If not, stop the cell and go back to your code!

(refer tens.py)

Expected output: although it may not match perfectly, your expected output should be close to ours and your cost value should decrease.

Cost after epoch 0 =	1.917929
Cost after epoch 5 =	1.506757
Train Accuracy =	0.940741
Test Accuracy =	0.783333
Congratulations! we have finised the assignment and built a model that recognizes SIGN language with almost 80% accuracy on the test set. we can actually improve its accuracy by spending more time tuning the hyperparameters, or using regularization (as this model clearly has a high variance).
