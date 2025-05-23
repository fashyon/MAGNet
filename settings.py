import os
import logging

# root_dir = './datasets/Rain200H'
root_dir = 'F:\data1/fangsiyan\dataset/rain100H_fenbie'
real_dir = './datasets/real'
log_dir = './logdir'
log_test_dir = './log_test/'
show_dir = './showdir'
model_dir = './models'
data_dir = os.path.join(root_dir, 'train/rain')
mat_files = os.listdir(data_dir)
num_datasets = len(mat_files)

lr = 5e-4
batch_size = 8
patch_size = 160
epoch = 600
aug_data = False
total_step = int((epoch * num_datasets)/batch_size)
one_epoch_step = int(num_datasets/batch_size)
save_steps = 50
save_epochs = 50

num_workers = 0
device_id = '0'

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


