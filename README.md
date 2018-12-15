# image2story

Written by LIN Jinghong, WEI Qing, SHI Siyuan

## project introduction

We set up a model for generating a story of some speciﬁc style from pictures and named it "image2story". The model is based on the previous image caption model and we expand the image caption model to generate a complete paragraph instead of a single sentence. Our model can be roughly divided into three parts. First, the sentence description is extracted from the picture through the image caption network, at the same time, the skip - thought vectors model is trained to realize the conversion between sentence and vector. Then we feed the sentences extracted by image caption into the encoder of skip - thought vectors, and convert the image caption into a vector
with its own text style. Then we use romantic novels as our data set and train a story generator, the principle of which is similar to that of skip - thought vectors. Finally, the vector is fed into the story generator to generate romantic paragraphs.

