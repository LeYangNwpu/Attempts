# parameters

N_EPOCHS = 200
MAX_PATIENCE = 30
batch_size = 16
N_CLASSES = 2
LEARNING_RATE = 1e-4
processor_num = 20
LR_DECAY = 0.995
DECAY_LR_EVERY_N_EPOCHS = 1
WEIGHT_DECAY = 0.0001


ori_train_base_rp = '/home/yangle/TCyb/dataset/cat_128_del/TrainPatch/'
ori_val_base_rp = '/home/yangle/TCyb/dataset/cat_128_del/ValPatch/'
res_root_path = '/home/yangle/TCyb/dataset/proc_8ch_128/'
EXPERIMENT = '/home/yangle/TCyb/result/TrainNet/'
EXPNAME = 'cat_8ch'
res_train_base_rp = res_root_path + 'train'
res_val_base_rp = res_root_path + 'val'



seed = 0
CUDNN = True
