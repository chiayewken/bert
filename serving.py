import tensorflow as tf
import os
import fire


def get_checkpoint_paths(output_dir):
  state = tf.train.get_checkpoint_state(output_dir)
  paths = state.all_model_checkpoint_paths
  tf.logging.info("Checkpoint paths: {}".format(paths))
  return paths


def remove_adam_vars(output_dir: str) -> None:
  # https://towardsdatascience.com/3-ways-to-optimize-and-export-bert-model-for-online-serving-8f49d774a501
  checkpoint = sorted(get_checkpoint_paths(output_dir))[-1]
  export_dir = os.path.join(output_dir, 'export')

  sess = tf.Session()
  imported_meta = tf.train.import_meta_graph(".".join([checkpoint, "meta"]))
  imported_meta.restore(sess, checkpoint)
  my_vars = []
  for var in tf.all_variables():
    if 'adam_v' not in var.name and 'adam_m' not in var.name:
      my_vars.append(var)
  saver = tf.train.Saver(my_vars)
  saver.save(sess, os.path.join(export_dir, 'model.ckpt'))
