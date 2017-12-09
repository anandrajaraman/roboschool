"""
Template script, ESE-680 Final Project.
"""
from OpenGL import GLU
import os, gym, roboschool
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
config = tf.ConfigProto(
    inter_op_parallelism_threads=1,
    intra_op_parallelism_threads=1,
    device_count = { "GPU": 0 } )
sess = tf.InteractiveSession(config=config)


def demo_run():
    """We're gonna do some cool stuff here."""


if __name__ == "__main__":
    demo_run()
