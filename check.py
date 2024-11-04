import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set to '3' for complete suppression
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
