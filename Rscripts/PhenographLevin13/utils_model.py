import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from utils_evaluation import plot3D_cluster_colors
from plotly.io import to_html

def frange_anneal(n_epoch, ratio=0.25, shape='sin'):
    L = np.ones(n_epoch)
    if ratio ==0:
        return L
    for c in range(n_epoch):
        if c <= np.floor(n_epoch*ratio):
            if shape=='sqrt':
                norm = np.sqrt(np.floor(n_epoch*ratio))
                L[c] = np.sqrt(c)/norm
            if shape=='sin':
                Om = (np.pi/2/(n_epoch*ratio))
                L[c] =  np.sin(Om*c)**4
        else:
            L[c]=1
    return L

class AnnealingCallback(Callback):
    def __init__(self, weight, kl_weight_lst):
        self.weight = weight
        self.kl_weight_lst = kl_weight_lst

    def on_epoch_end(self, epoch, logs={}):
        new_weight = K.eval(self.kl_weight_lst[epoch])
        K.set_value(self.weight, new_weight)
        print("  Current AP is " + str(K.get_value(self.weight)))

class EpochCounterCallback(Callback):
    def __init__(self, count, count_lst):
        self.count = count
        self.count_lst = count_lst

    def on_epoch_end(self, epoch, logs={}):
        new_count = K.eval(self.count_lst[epoch])
        K.set_value(self.count, new_count)
        #print("  Current AP is " + str(K.get_value(self.count)))

class plotCallback(Callback):
    def __init__(self, aFrame, Sigma, lbls, encoder, ID, bl, epochs, output_dir, save_period):
        super(Callback, self).__init__()
        self.aFrame = aFrame
        self.Sigma = Sigma
        self.lbls = lbls
        self.encoder = encoder
        self.ID = ID
        self.bl = bl
        self.epochs = epochs
        self.output_dir = output_dir
        self.save_period = save_period

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_period == 0 or epoch in range(0, 20):
            z = self.encoder.predict([self.aFrame, self.Sigma])
            fig = plot3D_cluster_colors(z, lbls=self.lbls)
            html_str = to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                               include_mathjax=False, post_script=None, full_html=True,
                               animation_opts=None, default_width='100%', default_height='100%', validate=True)
            html_dir = self.output_dir
            Html_file = open(html_dir + "/" + self.ID + "_" + str(self.bl) + 'epochs' + str(self.epochs) + '_epoch=' + str(
                epoch) + '_' + "_Buttons.html", "w")
            Html_file.write(html_str)
            Html_file.close()


class saveEncoder(Callback):
    def __init__(self,  encoder, ID, bl, epochs, output_dir, save_period):
        super(Callback, self).__init__()
        self.encoder = encoder
        self.ID = ID
        self.bl = bl
        self.epochs = epochs
        self.output_dir = output_dir
        self.save_period = save_period

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_period == 0 and epoch != 0:
            self.encoder.save_weights(
                self.output_dir + '/' + self.ID + "_encoder_" + str(self.bl) + 'epochs' + str(self.epochs) + '_epoch=' + str(
                    epoch) + '_3D.h5')


class relative_stop_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('DCAE_loss') < logs.get('mean_square_error_NN') / 10 and logs.get(
                'DCAE_loss') < 0.01):  # select the accuracy
            print("\n !!! early stopping !!!")
            self.model.stop_training = True






# relu
def relu_derivative(a):
    cond = tf.math.greater_equal(a, tf.constant(0.0))
    return tf.where(cond, tf.constant(1.0), tf.constant(0.0))

# elu
def elu_derivative(a):
    cond = tf.math.greater_equal(a, tf.constant(0.0))
    return tf.where(cond, tf.constant(1.0), a + 1)

# leaky relu
def leaky_relu_derivative(a):
    cond = tf.math.greater_equal(a, tf.constant(0.0))
    return tf.where(cond, tf.constant(1.0), tf.constant(0.3))

    # linear
def linear_derivative(a):
    return tf.ones(tf.shape(a), dtype=tf.float32)
