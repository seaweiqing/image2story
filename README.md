# image2story

Written by LIN Jinghong, WEI Qing, SHI Siyuan

## project introduction

We set up a model for generating a story of some speciﬁc style from pictures and named it "image2story". The model is based on the previous image caption model and we expand the image caption model to generate a complete paragraph instead of a single sentence. Our model can be roughly divided into three parts. First, the sentence description is extracted from the picture through the image caption network, at the same time, the skip - thought vectors model is trained to realize the conversion between sentence and vector. Then we feed the sentences extracted by image caption into the encoder of skip - thought vectors, and convert the image caption into a vector
with its own text style. Then we use romantic novels as our data set and train a story generator, the principle of which is similar to that of skip - thought vectors. Finally, the vector is fed into the story generator to generate romantic paragraphs.

## pretrained model
The pretrained model merges the image caption model from https://github.com/DeepRNN/image_captioning and a sentence-to-story model extracted from https://github.com/ryankiros/neural-storyteller.
First, download the need model [this](link),put the caption.npy into the caption_model folder, put the stv_model folder as well as romance_models folder into the story_model folder. Then put the image you would like to test into the test/images folder and run the demo.py the generated stories will be stored in the variable -passages. Also, it will produce the image caption images each with 3 captions, which is stored in the test/results folder.


## dependencies
 * Python 3
 * Theano
 * tensorflow
 
## output samples

![Result](https://github.com/seaweiqing/image2story/blob/master/output_samples/o_8.png)
![Result](https://github.com/seaweiqing/image2story/blob/master/output_samples/o_9.png)
![Result](https://github.com/seaweiqing/image2story/blob/master/output_samples/o_10.png)
![Result](https://github.com/seaweiqing/image2story/blob/master/output_samples/o_1.jpg)
![Result](https://github.com/seaweiqing/image2story/blob/master/output_samples/o_2.jpg)
![Result](https://github.com/seaweiqing/image2story/blob/master/output_samples/o_3.jpg)
![Result](https://github.com/seaweiqing/image2story/blob/master/output_samples/o_4.jpg)
![Result](https://github.com/seaweiqing/image2story/blob/master/output_samples/o_5.jpg)
![Result](https://github.com/seaweiqing/image2story/blob/master/output_samples/o_6.jpg)
