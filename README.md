# Abnormality Recognition in Medical Image Report

## Introduction
It is a NLP task in the medical field. Based on doctors' descriptions of CT scans, we need to build models to determine whether anomalies exist in specific areas of the patients' bodies.

## Data
For the reason of privacy protection, the text description has already replaced by the number,that is, each unique word is translated into the unique number. the label is just the ID of anomalies areas,if no any anomalies areas, then the label is empty.

## Modeling
For this task, I tried following classical deep learning models and the combination of them.
1.**Text-CNN**

2.**Bi-LSTM**

3.**Attention**

4.**Attention+Text-CNN**

5.**Bi-LSTM+Text-CNN**

6.**Attention+Bi-LSTM+Text-CNN**

Besides, I also compare the performance of each model on both the training set and validation set.Below shows the result.

![Training Loss](https://github.com/frankhjh/Abnormality-Recognition-in-Medical-Image-Report/blob/main/plot_out/Training%20Loss.png)

![Validation Loss](https://github.com/frankhjh/Abnormality-Recognition-in-Medical-Image-Report/blob/main/plot_out/Validation%20Loss.png)

## Try yourself
Feel free to use my model in `./model` for your own project.

If you want to run my code to train and make prediction by yourself,you can simply run the following command:

`python main.py --model_name ['cnn'/'rnn'/'attn'/'attn_cnn'/'lstm_cnn'/'attn_lstm_cnn'] --seed 1 --epochs 20 --device 'cpu'`.

Of course, you can change the parameters as you want,for the **--model_name** argument,you should choose one you want from the above list. For example 

`--model_name 'cnn'`

The final prediction result of test set will be stored in 
`./pred_out/submission.csv`



## Some References
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)

[LONG SHORT-TERM MEMORY BASED RECURRENT NEURAL NETWORK
ARCHITECTURES FOR LARGE VOCABULARY SPEECH RECOGNITION](https://arxiv.org/pdf/1402.1128.pdf)

[Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)