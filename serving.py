import six
import tensorflow as tf
import os
import fire


def get_checkpoint_paths(output_dir):
  state = tf.train.get_checkpoint_state(output_dir)
  paths = state.all_model_checkpoint_paths
  tf.logging.info("Checkpoint paths: {}".format(paths))
  return paths


def remove_adam_vars(output_dir: str, export_dir: str) -> None:
  # https://towardsdatascience.com/3-ways-to-optimize-and-export-bert-model-for-online-serving-8f49d774a501
  checkpoint = sorted(get_checkpoint_paths(output_dir))[-1]

  sess = tf.Session()
  imported_meta = tf.train.import_meta_graph(".".join([checkpoint, "meta"]))
  imported_meta.restore(sess, checkpoint)
  my_vars = []
  for var in tf.all_variables():
    if 'adam_v' not in var.name and 'adam_m' not in var.name:
      my_vars.append(var)
  saver = tf.train.Saver(my_vars)
  saver.save(sess, os.path.join(export_dir, 'model.ckpt'))


def clean_ckpt(input_ckpt, output_model_dir):
  tf.reset_default_graph()

  var_list = tf.contrib.framework.list_variables(input_ckpt)
  var_values, var_dtypes = {}, {}
  for (name, shape) in var_list:
    if not name.startswith("global_step") and "adam" not in name.lower():
      var_values[name] = None
      tf.logging.info("Include {}".format(name))
    else:
      tf.logging.info("Exclude {}".format(name))

  tf.logging.info("Loading from {}".format(input_ckpt))
  reader = tf.contrib.framework.load_checkpoint(input_ckpt)
  for name in var_values:
    tensor = reader.get_tensor(name)
    var_dtypes[name] = tensor.dtype
    var_values[name] = tensor

  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    tf_vars = [
        tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v])
        for v in var_values
    ]
  placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
  assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
  global_step = tf.Variable(0,
                            name="global_step",
                            trainable=False,
                            dtype=tf.int64)
  saver = tf.train.Saver(tf.all_variables())

  if not tf.gfile.Exists(output_model_dir):
    tf.gfile.MakeDirs(output_model_dir)

  # Build a model consisting only of variables, set them to the average values.
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                           six.iteritems(var_values)):
      sess.run(assign_op, {p: value})

    # Use the built saver to save the averaged checkpoint.
    saver.save(sess,
               os.path.join(output_model_dir, "model.ckpt"),
               global_step=global_step)


if __name__ == "__main__":
  # fire.Fire(remove_adam_vars)
  fire.Fire(clean_ckpt)
