# README

## 1. Background

---

In today's society, emotion recognition technology plays an increasingly important role in the field of speech processing. Understanding and recognizing the speaker's emotional state not only helps to improve the human-computer interaction experience, but also has practical application value in many application scenarios, such as customer service and virtual assistant. This project aims to build a comprehensive emotion recognition and generation system by integrating SER (Speech Emotion Recognition) technology, PaddleSpeech and EmotiVoice.

### 1.1 **Project Objectives**

- Training of Emotion Recognition Model: Using the RADVSS dataset, we trained an emotion recognition model using the SER technique. This model is able to recognize the speaker's emotional state from the input speech, covering a wide range of emotional categories such as happiness, anger, sadness and joy.
- Speech to text: Using the compiled PaddleSpeech, we can quickly and accurately convert the input speech into corresponding text information, which provides the basis for subsequent processing.
- Emotion Adjustment and Speech Generation: Using EmotiVoice technology, we are able to adjust the emotion based on the output of the emotion recognition model and generate speech output that matches the specified emotion. This step provides users with a more personalized and emotionally rich interactive experience.

### 1.2 **Technical Background**

- **SER TECHNOLOGY**: Speech Emotion Recognition is one of the key technologies in the field of speech processing, where a model is trained to recognize the emotional state in the speaker's voice. In this project, we train the RADVSS dataset to achieve accurate recognition of speech emotion.
- **PaddleSpeech**: PaddleSpeech is a speech processing tool based on the PaddlePaddle framework, which is able to realize the function of speech to text. We utilize this tool to provide a speech-to-text base service for our system.
- **EmotiVoice**: EmotiVoice is an emotionally synthesized speech generation tool that adjusts the pitch of the voice, the speed of speech, and other parameters to make the generated speech closer to the specified emotional state. By integrating EmotiVoice, we are able to realize more detailed emotional changes in the voice output.

Through the combination of the above technologies, this project aims to build a comprehensive and efficient speech emotion recognition and generation system to provide more intelligent and personalized solutions for applications in the field of speech processing.



## 2. Install

### 2.1 **Environments**

Make sure your system meets the following basic requirements:

- Python 3.8
- Keras & TensorFlow 2
- gcc >= 4.8.5
- paddlepaddle <= 2.5.1
- OS support: Linux(recommend), Windows, Mac OSX

### 2.2 **Requirments**

**Python**

- TensorFlow 2 / Keras：LSTM & CNN (`tensorflow.keras`)
- scikit-learn：SVM & MLP model, dividing the training and test sets
- joblib：Saving and loading models trained with scikit-learn
- librosa：Extracted features, waveform graphs
- SciPy：spectrogram
- pandas：Loading Features
- Matplotlib：drawings
- NumPy

**Tools**

- [Optional] Opensmile：Extraction Characteristics

### 2.3 Datasets

**RAVDESS**

English, approximately 1500 audios from 24 individuals (12 male, 12 female) expressing 8 different emotions (the third digit indicates the emotion category)：01 = neutral，02 = calm，03 = happy，04 = sad，05 = angry，06 = fearful，07 = disgust，08 = surprised.



## 3. Usage

### 3.1 Speech emotion recognition

**First clone from github**：

```
git clone https://github.com/Renovamen/Speech-Emotion-Recognition.git
```

**Installation of dependencies**:

```
pip install -r requirements.txt
```

(Optional) Install Opensmile.(Compilation needs to be done after git clone opensmile)

**Preprocess**

First you need to extract the features of the audio in the dataset and save them locally. features extracted by Opensmile will be saved in `.csv` files and features extracted by librosa will be saved in `.p` files.

```
python preprocess.py --config configs/example.yaml
```

where `configs/example.yaml` is the path to your configuration file.

**Train**

The dataset path can be configured in `configs/`, with audio of the same sentiment in the same folder

```
└── datasets
    ├── angry
    ├── happy
    ├── sad
    ...
```

then:

```
python train.py --config configs/example.yaml
```

**Predict**:

Use a trained model to predict the sentiment of a given audio:

```
python predict.py --config configs/example.yaml
```

Here we are using `opensmile` to extract the speech features and using `cnn1d` to train the resulting model:

```
└── checkpoints
    ├── CNN1D_OPENSMILE_IS10.h5
    ├── CNN1D_OPENSMILE_IS10.json
    ├── ...
    ...
```

The yaml training parameters used are as follows:

```
# training parameter
epochs: 100  # train epoch number
batch_size: 32  
lr: 0.001  # learn rate

# model parameter
n_kernels: 32  
kernel_sizes: [5, 5]  
dropout: 0.5
hidden_size: 32
```





### 3.2 **PaddleSpeech**

There are two quick installation methods for PaddleSpeech, one is pip installation, and the other is source code compilation,We have chosen to use source compilation here:

**source compilation**:

```
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
cd PaddleSpeech
pip install pytest-runner
pip install .
```



Then we can call its python library function to use the speech-to-text function.

```python
from PaddleSpeech.paddlespeech.cli.asr.infer import ASRExecutor
    asr = ASRExecutor()
    result = asr(audio_file=audio_file)
    print("您的识别结果是"+result)
```



### 3.3 EmotiVoice

**Full installation**:

```
conda create -n EmotiVoice python=3.8 -y
conda activate EmotiVoice
pip install torch torchaudio
pip install numpy numba scipy transformers soundfile yacs g2p_en jieba pypinyin
```

**Prepare model files**:

```
git lfs install
git lfs clone https://huggingface.co/WangZeJun/simbert-base-chinese WangZeJun/simbert-base-chinese
```

**inference**:

Download the pre-trained model by simply running the following command:

```
git clone https://www.modelscope.cn/syq163/outputs.git
```



### 3.4 Our Programs

In order to simplify the use, we wrote a python script file, the user can run the script, add the need to recognize the location of the voice file, as well as the subsequent need to generate the emotion of the keyword parameters,.

```
python detect.py audio_file_path emotion speaker
```

It can automatically call the appropriate commands to complete the entire process, and the final location of the generated file in the `output` folder of the same level as running the python script:

```
└── detect.py
└── output
	├── 1.wav
...
```

After running, it will automatically call the model of emotion recognition, and after getting the recognition result, it will call PaddleSpeech to recognize the speech and convert it to text, and finally generate the corresponding speech according to the input emotion parameter.

We offer the following emotions, but of course you can also customize the mood parameter inputs:

+ HAPPY
+ SAD
+ ANGRY
+ SURPRISED
+ NEUTRAL
+ FEARFUL
+ DISGUSTED 
+ CALM 

+ ...

Last but not least, we also provide a rich variety of tones for users to choose from, and the following tones are now available:

+ youngwoman
+ youngman
+ smallgirl
+ Middlemale

You can add the relevant tone parameter to the call to the python script.



## 4. Related Efforts

Speech-Emotion-Recognition:https://github.com/Renovamen/Speech-Emotion-Recognition.git

PaddleSpeech:https://github.com/PaddlePaddle/PaddleSpeech.git

EmotiVoice:https://github.com/netease-youdao/EmotiVoice.git





