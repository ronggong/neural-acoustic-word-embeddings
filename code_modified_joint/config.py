from os import path, makedirs


class Config(object):
    """Set up model for debugging."""

    def __init__(self, margin_input, output_shape, num_layers, mtl, folder_model):
        self.margin = margin_input
        self.logdir = path.join("../logs/", folder_model)
        self.ckptdir = path.join("../ckpts/", folder_model)
        makedirs(self.logdir, exist_ok=True)
        makedirs(self.ckptdir, exist_ok=True)
        self.output_shape = output_shape  # output list multitask learning [27, 2] is [phn, professionality]
        self.num_layers = num_layers
        self.mtl = mtl # ["phn", "pro", "both"]

    # path_dataset = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/'
    path_dataset = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/RongGong/phoneEmbedding'

    filename_feature_teacher = path.join(path_dataset, 'feature_phn_embedding_train_teacher.pkl')
    filename_list_key_teacher = path.join(path_dataset, 'list_key_teacher.pkl')
    filename_feature_student = path.join(path_dataset, 'feature_phn_embedding_train_student.pkl')
    filename_list_key_student = path.join(path_dataset, 'list_key_student.pkl')

    filename_scaler = path.join(path_dataset, 'scaler_phn_embedding_train_teacher_student.pkl')
    filename_label_encoder = path.join(path_dataset, 'le_phn_embedding_teacher_student.pkl')
    filename_data_splits = path.join(path_dataset, 'data_splits_teacher_student.pkl')

    # validation dataset
    filename_feature_teacher_val = path.join(path_dataset, 'feature_phn_embedding_val_teacher.pkl')
    filename_feature_student_val = path.join(path_dataset, 'feature_phn_embedding_val_student.pkl')

    # test dataset
    filename_feature_teacher_test = path.join(path_dataset, 'feature_phn_embedding_test_teacher.pkl')
    filename_feature_student_test = path.join(path_dataset, 'feature_phn_embedding_test_extra_student.pkl')
    filename_list_key_student_test = path.join(path_dataset, 'list_key_extra_student.pkl')

    batch_size = 128
    current_epoch = 0
    num_epochs = 500
    feature_dim = 80
    hidden_size = 32
    bidirectional = True
    keep_prob = 1.0
    max_same = 1
    max_diff = 1
    lr = 0.001
    mom = 0.9
    log_interval = 10
    early_stopping_step = 15
    ckpt = None
    debugmode = True