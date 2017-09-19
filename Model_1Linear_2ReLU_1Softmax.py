# ---------------------------------------------------------------    
# Linear layer:
def LinearLayer(x,dim1,dim2):
    import tensorflow as tf
    # the W & b here are random normal (before training)
    W=tf.Variable(tf.random_normal([dim1,dim2],stddev=0.01))
    b=tf.Variable(tf.random_normal([1, dim2],stddev=0.01))
    return tf.matmul(x,W)+b

# ---------------------------------------------------------------    
# ReLU hidden layer: (one linear layer inside)
# ReLU function itself wouldn't change dimension of data.
# y = max(0, x)
def ReLULayer(x,dim1,dim2,dropout_prob=0.5):
    import tensorflow as tf
    W_h = tf.Variable(tf.truncated_normal([dim1,dim2], stddev=0.1))
    b_h = tf.constant(0.1, shape=[dim2])
    z_h = tf.matmul(x,W_h) + b_h
    y_h = tf.nn.relu(z_h) 
    
    # Dropout layer
    output = tf.nn.dropout(y_h, dropout_prob)
    return output


# Construct Model has 2 hidden ReLU layers.
def Model(x,dim1,dim2, n_hidden):
    import tensorflow as tf
    # apply dropout to input layer
    prob_keep_input=0.8
    x=tf.nn.dropout(x,prob_keep_input)
    
    dimH1, dimH2=n_hidden, n_hidden
    
    #hidden layer 1:
    h1=ReLULayer(x,dim1,dimH1,dropout_prob=0.5)
    #hidden layer 2:
    h2=ReLULayer(h1,dimH1,dimH2,dropout_prob=0.5)
    
    # output layer (Sigmoid layer):
    # Sigmoid layer doesm't change dimension of data.
    logits=LinearLayer(h2,dimH2,dim2)
#     logits=LinearLayer(h2,dimH2,dim2)
    y_pred_prob=tf.nn.softmax(logits)
    # y_ is the predicted label
    
    return [logits, y_pred_prob]


