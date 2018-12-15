# model overview

Our model can be divided into three parts: ﬁrst, we use image caption to get the sentence description of the input picture, then we train the model of skip-thought vectors to convert a sentence into a vector, ﬁnally, we feed the sentence vector obtained by image caption into our story generator, during this plug-in process, we subtract the mean of image caption skip-thought vectors and then feed it into our story generator, thus, the ﬁnal output will only contain our romance novel style.

[!model overview](https://github.com/seaweiqing/image2story/blob/master/model_overview/dl_1.png)
[!model overview](https://github.com/seaweiqing/image2story/blob/master/model_overview/dl_2.png)
[!model overview](https://github.com/seaweiqing/image2story/blob/master/model_overview/dl_3.png)
[!model overview](https://github.com/seaweiqing/image2story/blob/master/model_overview/dl_4.png)
[!model overview](https://github.com/seaweiqing/image2story/blob/master/model_overview/dl_5.png)
[!model overview](https://github.com/seaweiqing/image2story/blob/master/model_overview/dl_6.png)
