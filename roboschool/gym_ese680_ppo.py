model_meta_fn = 'FILENAME.cpkt.meta'

import tensorflow as tf

sess = tf.Session()
saver = tf.train.import_meta_graph(model_meta_fn)
saver.restore(sess, tf.train.latest_checkpoint('.'))
graph = tf.get_default_graph()

print("Model loaded")


def get_var_by_name(desired):
    return [v for v in tf.global_variables() if v.name == desired][0]


state_mean = get_var_by_name('pi/obfilter/runningsum:0') / get_var_by_name('pi/obfilter/count:0')
state_std = tf.sqrt(tf.maximum(
    get_var_by_name('pi/obfilter/runningsumsq:0') / get_var_by_name('pi/obfilter/count:0') - tf.square(state_mean),
    1e-2))
policy = {'state_means': sess.run(state_mean),
          'state_std': sess.run(state_std),
          'w0': sess.run(get_var_by_name('pi/polfc1/w:0')),
          'b0': sess.run(get_var_by_name('pi/polfc1/b:0')),
          'w1': sess.run(get_var_by_name('pi/polfc2/w:0')),
          'b1': sess.run(get_var_by_name('pi/polfc2/b:0')),
          'w2': sess.run(get_var_by_name('pi/polfinal/w:0')),
          'b2': sess.run(get_var_by_name('pi/polfinal/b:0'))
          }


def recursive_np_array_print(a, idx):
    if len(idx) == len(a.shape) - 1:
        return "[" + ",".join(["%+8.4f" % a[tuple(idx + [i])] for i in range(a.shape[len(idx)])]) + "]"
    else:
        return "[\n" + ",\n".join([recursive_np_array_print(a, idx + [i]) for i in range(a.shape[len(idx)])]) + "\n]"


str = "import numpy as np\n"
for key in policy:
    str += key + " = np.array(" + recursive_np_array_print(policy[key], []) + ")\n"

file = open('model.weights', "w")
file.write(str)
file.close()

print("Policy weights dumped to model.weights")
