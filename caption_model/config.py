class Config(object):
    """ Wrapper class for various (hyper)parameters. """

    def __init__(self):
        # about the model architecture
        self.cnn = 'vgg16'  # 'vgg16' or 'resnet50'
        self.max_caption_length = 20
        self.dim_embedding = 512
        self.num_lstm_units = 512
        self.num_initalize_layers = 2  # 1 or 2
        self.dim_initalize_layer = 512
        self.num_attend_layers = 2  # 1 or 2
        self.dim_attend_layer = 512
        self.num_decode_layers = 2  # 1 or 2
        self.dim_decode_layer = 1024

        # about the weight initialization and regularization
        self.fc_kernel_initializer_scale = 0.08
        self.fc_kernel_regularizer_scale = 1e-4
        self.fc_activity_regularizer_scale = 0.0
        self.conv_kernel_regularizer_scale = 1e-4
        self.conv_activity_regularizer_scale = 0.0
        self.fc_drop_rate = 0.5
        self.lstm_drop_rate = 0.3
        self.attention_loss_factor = 0.01

        # about the optimization
        self.num_epochs = 100
        self.batch_size = 1
        self.optimizer = 'Adam'  # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
        self.initial_learning_rate = 0.0001
        self.learning_rate_decay_factor = 1.0
        self.num_steps_per_decay = 100000
        self.clip_gradients = 5.0
        self.momentum = 0.0
        self.use_nesterov = True
        self.decay = 0.9
        self.centered = True
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6

        # about the vocabulary
        self.vocabulary_file = './caption_model/vocabulary.csv'
        self.vocabulary_size = 5000
        self.train_caption_file = './caption_model/captions_train2014.json'

        # about the testing
        self.test_image_dir = './test/images/'
        self.test_result_dir = './test/results/'

        # --------self-define--------- #
        self.phase = 'test'
        self.train_cnn = False
        self.beam_size = 3
        self.model_file = './caption_model/caption.npy'
