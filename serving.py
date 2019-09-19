import tensorflow as tf
import os
import fire


def remove_adam_vars(ckpt_path: str) -> None:
  # https://towardsdatascience.com/3-ways-to-optimize-and-export-bert-model-for-online-serving-8f49d774a501
  #path that contains all 3 ckpt files of your fine-tuned model
  # path = './bert_output'
  path = ckpt_path

  #path to output the new optimized model
  output_path = os.path.join(path, 'optimized_model')

  sess = tf.Session()
  imported_meta = tf.train.import_meta_graph(
      os.path.join(path, 'model.ckpt-236962.meta')
  )  #based on the steps of your fine-tuned model
  imported_meta.restore(sess, os.path.join(
      path, 'model.ckpt-236962'))  #based on the steps of your fine-tuned model
  my_vars = []
  for var in tf.all_variables():
    if 'adam_v' not in var.name and 'adam_m' not in var.name:
      my_vars.append(var)
  saver = tf.train.Saver(my_vars)
  saver.save(sess, os.path.join(
      output_path,
      'model.ckpt'))  #change model.ckpt to name of your preference
