import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from datetime import datetime

#########################################
# x = tf.Variable(3, name="x")
# y = tf.Variable(4, name="y")
# f = x * x * y + y + 2
#
# with tf.Session() as sess:
#     x.initializer.run()
#     y.initializer.run()
#     result = f.eval()
# print(result)
#########################################

# w = tf.constant(3)
# x = w + 2
# y = x + 5
# z = x * 3
#
# with tf.Session() as sess:
#     print(y.eval())  # 10
#     print(z.eval())  # 15

#########################################

# housing = fetch_california_housing()
# m, n = housing.data.shape
# housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
#
# X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
#
# XT = tf.transpose(X)
# theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
#
# with tf.Session() as sess:
#     theta_value = theta.eval()
# print(theta_value)

#########################################
housing = fetch_california_housing()
np.set_printoptions(suppress=True)

m, n = housing.data.shape
# housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
# print("Housing data: \n", housing_data_plus_bias[0])

# from sklearn.preprocessing import normalize
# scaled_housing_data_plus_bias = housing_data_plus_bias / np.linalg.norm(housing_data_plus_bias)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
print("Scaled housing data: \n", scaled_housing_data[0])

# Adding bias after scaling done
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
print("Scaled housing data plus bias: \n", scaled_housing_data_plus_bias[0])

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y

mse = tf.reduce_mean(tf.square(error), name="mse")

# Manual gradient computation
# gradients = (2/m) * tf.matmul(tf.transpose(X), error)

# Auto gradient computation
# gradients = tf.gradients(mse, [theta])[0]

# Manual optimization with gradient
# training_op = tf.assign(theta, theta - learning_rate * gradients)

# Auto optimizer
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) # Gradient optimizer
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9) # Momentum optimizer
training_op = optimizer.minimize(mse)

### Comment for turn-off batch gradient descent \/
# X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
# y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
# batch_size = 100
# n_batches = int(np.ceil(m / batch_size))
# def fetch_batch(epoch, batch_index, batch_size):
#     np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
#     indices = np.random.randint(m, size=batch_size)  # not shown
#     X_batch = scaled_housing_data_plus_bias[indices]  # not shown
#     y_batch = housing.target.reshape(-1, 1)[indices]  # not shown
#     return X_batch, y_batch
### Comment for turn-off batch gradient descent /\

init = tf.global_variables_initializer()
saver = tf.train.Saver()

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

with tf.Session() as sess:
    sess.run(init)

    print("Theta init: \n", theta.eval())
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
            save_path = saver.save(sess, "/tmp/my_model.ckpt")
        sess.run(training_op)


    ### Comment for turn-off batch gradient descent \/
    # for epoch in range(n_epochs):
    #     if epoch % 100 == 0:
    #         print("Epoch", epoch, "MSE =", mse.eval())
    #     for batch_index in range(n_batches):
    #         X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
    #         sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    ### Comment for turn-off batch gradient descent /\
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")
    best_theta = theta.eval()


####################### Placeholders
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})

# print(B_val_1)
# print(B_val_2)
#######################