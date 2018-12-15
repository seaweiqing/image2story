import numpy as np
import tensorflow as tf
from caption_model.config import Config
from caption_model.dataset import prepare_test_data
from caption_model.model import CaptionGenerator
from story_model import skipthoughts, decoder

config = Config()
# ------test--------#
tf.reset_default_graph()

with tf.Session() as sess:
    model = CaptionGenerator(config)
    model.load(sess, config.model_file)
    tf.get_default_graph().finalize()
    data, vocabulary = prepare_test_data(config)
    info = model.test(sess, data, vocabulary)

# story
path = './story_model/stv_model/'
encoder = skipthoughts.load_model(path, path)
decode = decoder.load_model('./story_model/romance_models/romance.npz',
                            './story_model/romance_models/romance_dictionary.pkl')
bneg = np.load('./story_model/romance_models/caption_style.npy')
bpos = np.load('./story_model/romance_models/romance_style.npy')
passages = []
for num in range(len(info)):
    sentence = info[num]['cap']
    # Compute skip-thought vectors for sentences
    svecs = skipthoughts.encode(encoder, sentence, verbose=False)
    # Style shifting
    shift = svecs.mean(0) - bneg + bpos
    passage = decoder.run_sampler(decode, shift, beam_width=3, maxlen=200)
    image_file = info[num]['img_path']
    passages.append({'img': image_file, 'passage': passage})
    print('done:%d' % num)

import matplotlib.pyplot as plt
img = plt.imread('./test/images/test1.jpg')
img.shape