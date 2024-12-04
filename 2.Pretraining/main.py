import pretrain
import evaluation
import tensorflow as tf
import os

os.environ["PATH"] += os.pathsep + r"C:/Program Files/Graphviz/bin"

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
'''strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))'''

#   2018 -> 559, 563, 570, 575, 588, 591
#   2020 -> 540, 544, 552, 567, 584, 596
#   openAPS -> 1, 2, 3

patients = [1,2,3,4,5,6]
rescaling = False

pretrain.main(rescaling=rescaling)  # Dataset without 2018 data
evaluation.main(raw_evaluation=True, year=2018, patients=patients, rescaling=rescaling)
#preevaluation.main(year=0)
