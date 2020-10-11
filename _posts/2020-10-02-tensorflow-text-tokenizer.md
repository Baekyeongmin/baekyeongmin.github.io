---
title: "Tensorflow-text로 Sentencepiece 토크나이저 이용하기"
layout: post
categories:
  - dev
tags:
  - dev
last_modified_at: 2020-10-02T20:53:50-05:00
author: yeongmin
comments: true
---

이번 포스트에서는 [`tensorflow-text`](https://github.com/tensorflow/text), [`tf_sentencepiece`](https://github.com/google/sentencepiece/tree/master/tensorflow) 모듈을 이용하여 학습 코드 상에서 토크나이징을 진행하고, 결과 모델을 export하는 과정까지의 경험을 기록한다. `tensorflow-text`는 텍스트 전처리 과정의 연산들을 Tensorflow 그래프 상에포함할 수 있도록 해주고, `tf-sentencepiece`는 자연어 처리에서 자주 이용되는 Sentencepiece 토크나이저를 `tensorflow-text` 토크나이저 형식에 맞춰 쉽게 이용할 수 있도록 해준다. 

학습 측면에서 살펴보면, tensorflow를 이용하여 학습을 진행하는 과정은 1) 데이터 전처리 및 TFRecord 파일 생성 2) 1에서 생성된 파일들을 이용하여 학습의 두 단계로 구성된다. 이전에는 TFRecord를 만들 때, 모든 데이터들을 토크나이징하고 인덱싱하여 저장했지만, `tensorflow-text`를 이용하면 TFRecord를 String으로 저장하고 이를 읽어서 바로 토크나이징하여 학습에 이용할 수 있다. 

서빙 측면에서 보면, 일반적으로 Tensorflow로 학습된 모델을 `SavedModel` 형태로 저장하고 이를 `tf-serving`, `TFLite` 등 을 이용하여 서빙에 이용한다. 많은 NLP 모델들은 Sentencepiece, WordPiece 등의 토크나이저를 이용하는데, 이를 서빙하기 위해 모델 서버에 요청을 보내기 전에 별도의 토크나이징 + 인덱싱 과정이 필요했다. `tensorflow-text`를 이용하면 모델 서버에 바로 텍스트로 요청을 보내고, 이를 처리할 수 있다.

<br>

# 1. Tensorflow-text

[Tensorflow-text 공식문서](https://github.com/tensorflow/text) 설명을 읽어보면, Tensorflow 코어에서는 지원하지 않는 텍스트 피쳐를 다루기 위한 유용한 기능들을 제공하는 라이브러리이다.(Tensorflow2.x 버전부터 이용이 가능한 것 같다.) 설치는 pip을 이용해 간단히 할 수 있으므로 생략한다.(단 tensorflow와 tensorflow-text의 마이너 버전은 맞춰줘야 한다.) 위에서 잠깐 언급했지만, 이 모듈의 가장 큰 장점은 텍스트 전처리 과정을 Tensorflow graph 상에 포함시킬 수 있다는 점이다. 이를 통해 전처리 스크립트를 관리하기 쉽고, 학습과 추론시에 완전히 동일한 전처리 과정을 보장할 수 있다. Unicode, Normalization 등 다양한 기능들을 제공하지만, 이번 글에서는 Tokenization에 집중한다.

## Tokenization

토크나이저는 입력 텍스트를 `토큰`의 단위로 잘라준다. 가장 간단하게는 띄어쓰기나 캐릭터 단위로 분리할 수 있다. 최근에는 Sentencepiece나 WordPiece 등의 토크나이저가 주로 이용된다. Tensorflow-text에서는 기본적으로 `WhitespaceTokenizer`와 `UnicodeScriptTokenizer`를 제공한다. `WhitespaceTokenizer`는 띄어쓰기 단위로 토크나이징을 하고, `UnicodeScriptTokenizer`는 띄어쓰기와 유사하지만 띄어쓰기 이외에 몇몇 Unicode를 기준으로 토크나이징을 한다. 아래 예제와 같이, 선언을 하고 `.tokenize()` method를 이용하면 손쉽게 텍스트를 자를 수 있다.

```python
import tensorflow_text as tf_text

tokenizer = tf_text.UnicodeScriptTokenizer()
tokens = tokenizer.tokenize(['everything not saved will be lost.',
                             u'Sad☹'.encode('UTF-8')])
print(tokens.to_list())
```
```
[['everything', 'not', 'saved', 'will', 'be', 'lost', '.'],
 ['Sad', '\xe2\x98\xb9']]
```

<br>

# 2. tf-sentencepiece

이 모듈은 구글의 [Sentencepiece 공식 구현체](https://github.com/google/sentencepiece/tree/master/tensorflow)에 구현되어 있으며, pip으로 쉽게 설치할 수 있다. Sentencepiece의 설명은 [이 논문](https://arxiv.org/abs/1808.06226)을 참고할 수 있고, 위 레포에는 BPE, Unigram-LM등 다양한 알고리즘들이 구현되어 있다. 학습을 진행할 텍스트 파일만 있으면 몇 줄 안되는 코드로 쉽게 토크나이저를 학습하고, 불러올 수 있다. 자세한 과정은 생략하고, 학습된 sentencepiece 모델 파일(`.model`)이 있다고 가정하고 이후 단계를 진행한다.

## Sentencepiece 모델 불러오기(기존 방법)

일반적으로 python에서 학습된 Sentencepiece 모델을 불러오고, tokenize하는 과정은 다음과 같다. tokenize는 `.encode()` method로 할 수 있으며, `out_type` 인자가 `str`인 경우 잘려진 토큰들의 리스트가 `int`인 경우 잘려진 토큰들의 사전 index의 리스트가 반환된다.

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file=MODEL_FILE_PATH)
print(sp.encode("토크나이저 테스트", out_type=str))
print(sp.encode("토크나이저 테스트", out_type=int))
```

```python
['▁', '토크', '나', '이', '저', '▁테스트']
[3, 14338, 30, 7, 512, 13167]
```

## Sentencepiece 모델 불러오기(tensorflow-text)

`tensorflow-text` 에서 학습된 Sentencepiece 모델을 불러오고, tokenize하는 과정은 다음과 같다. 위 과정과 유사하고, `SentencepieceTokenizer`를 초기화 할 때, `out_type` 인자를 조정하여 출력 값의 타입을 설정할 수 있다. 출력 타입은 `tf.RaggedTensor`인데, 토크나이저에서 처음 접했고 [공식문서](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor)를 참조하면 하나 이상의 차원에서 각각의 원소가 다른 길이를 갖는 텐서이다. 토크나이징 결과의 길이는 입력 문장에 따라 달라지기 때문에 위 텐서타입을 반환한다. 결과는 기존 방법과 동일함을 확인할 수 있다.

```python
import tensorflow as tf
import tensorflow_text as tf_text

model = open(f"{MODEL_PREFIX}.model", "rb").read()
tensorflow_sp_out_int = tf_text.SentencepieceTokenizer(model=model)
tensorflow_sp_out_str = tf_text.SentencepieceTokenizer(model=model, out_type=tf.string)
print(tensorflow_sp_out_str.tokenize(["토크나이저 테스트"]))
print(tensorflow_sp_out_int.tokenize(["토크나이저 테스트"]))
```

```python
<tf.RaggedTensor [[b'\xe2\x96\x81', b'\xed\x86\xa0\xed\x81\xac', b'\xeb\x82\x98', b'\xec\x9d\xb4', b'\xec\xa0\x80', b'\xe2\x96\x81\xed\x85\x8c\xec\x8a\xa4\xed\x8a\xb8']]>
<tf.RaggedTensor [[3, 14338, 30, 7, 512, 13167]]>
```

<br>

# 3. NSMC 데이터를 이용한 실습

[NSMC(Naver Sentiment Movie Corpus)](https://github.com/e9t/nsmc) 를 이용해 Tokenizing 과정이 포함된 간단한 모델을 학습하고, 학습된 모델을 Export하는 과정 까지 진행한다. 모든 스크립트는 [이 저장소]()의 코드를 이용한다. 

## 3.1. 데이터 전처리

데이터는 위 링크의 `ratings_train.txt`, `ratings_test.txt` 두 개의 파일을 이용하고, 각 파일은 `id`	,`document`,`label`로 구성된다. `label`은 해당 영화 리뷰가 긍정적인지(1), 부정적인지(0)로 저장되어 있다. 일반적인 tensorflow 학습은 데이터를 TFRecord 형식으로 저장하고, 이를 이용한다. [Tensorflow 공식 튜토리얼](https://www.tensorflow.org/tutorials/load_data/tfrecord) 을 참고하여 간단하게 NSMC 데이터를 읽어서 TFRecord 형태로 저장하는 코드는 다음과 같다.

```python
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(text, label):
    feature = {
        'text': _bytes_feature(text),
        'label': _int64_feature(label),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

with open(TRAIN_FILE_PATH) as f:
    with tf.io.TFRecordWriter(TRAIN_TF_RECORD_PATH) as writer:
        for line in f.readlines()[1:]:
            text, label = line.strip("\n").split("\t")[1:]
            example = serialize_example(text.encode("utf-8"), int(label))
            writer.write(example)
```

## 3.2. 토크나이징 과정을 포함한 모델

`tf-text`를 이용하면 토크나이징 과정을 Tensorflow Graph 연산에 포함시킬 수 있다. 이번 포스트에서는 간단한 구현을 위해 `tensorflow.keras.Model`의 `.call()` 메소드 내부에 토크나이징을 포함한다. 명시적으로 전처리 과정을 구분하고 싶다면, 토크나이징 부분을 밖으로 빼고, `@tf.function`의 형태로 구현하여 그래프에 포함시킬 수도 있다. 

모델은 Tokenizing → Embedding → BiLSTM → Dense 레이어 순으로 구성되며, 최종적으로 두 개의 logit을 출력한다. 위에서 잠깐 언급했듯이, `tf_text.Tokenizer`의 `.tokenize()` 메소드는 `tf.RaggedTensor`를 반환하는데, 이 텐서 타입의 `to_tensor()` 메소드를 이용하면 [`batch_size`, `sequnece_length`] 형태의 Dense 텐서를 얻을 수 있다. (Dense 텐서로 변환하면 이후 연산을 진행할 수 있다.)

```python
class SimpleTextClassifier(tf.keras.Model):
    def __init__(self, 
                 tokenizer_path,
                 vocab_size, 
                 hidden_size, 
                 output_size, 
                 default_value=0, 
                 max_sequence_length=128,
                 *args, 
                 **kwargs):
        super(SimpleTextClassifier, self).__init__(*args, **kwargs)
        
        self.default_value = default_value
        self.max_sequence_length = max_sequence_length
        
        self.tokenizer = tf_text.SentencepieceTokenizer(model=open(tokenizer_path, "rb").read())
        self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_size)
        self.lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size))
        self.output_layer = tf.keras.layers.Dense(output_size)
    
    def call(self, inputs, training=False):
        # Tokenizing
        tokenized_inputs = self.tokenizer.tokenize(inputs).to_tensor(
            default_value=self.default_value, 
            shape=[None, self.max_sequence_length]
        )

        sequence_mask = tokenized_inputs != self.default_value
        embedding = self.embedding(tokenized_inputs, training=training)
        encoded_output = self.lstm_layer(embedding, mask=sequence_mask, training=training)
        output = self.output_layer(encoded_output, training=training)
        return output

```

## 3.3. 학습

[Tensorflow Customized Training Loop 튜토리얼](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough)을 참고하여 아래와 같은 학습코드를 구성할 수 있다.

```python
def forward_step(batch, model, loss_fn, metrics, training=False):
    output = model(batch["text"], training=training)
    label = tf.one_hot(batch["label"], 2)
    loss = loss_fn(label, output)
    argmax_output = tf.argmax(output, -1)
    
    metrics["accuracy"].update_state(batch["label"], argmax_output)
    metrics["precision"].update_state(batch["label"], argmax_output)
    metrics["recall"].update_state(batch["label"], argmax_output)
    metrics["loss"].update_state(loss)
    
    return loss

@tf.function
def train_step(batch, model, optimizer, loss_fn, metrics):
    with tf.GradientTape() as tape:
        loss = forward_step(batch, model, loss_fn, metrics, True)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function
def valid_step(batch, model, loss_fn, metrics):
    forward_step(batch, model, loss_fn, metrics, False)
  
model = SimpleTextClassifier(SPM_MODEL_PATH, 16000, 128, 2)
optimizer = tf.keras.optimizers.Adam(1e-3)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
checkpoint = tf.train.Checkpoint(model=model)
ckpt_manager = tf.train.CheckpointManager(checkpoint, MODEL_SAVE_PATH, max_to_keep=5)

for idx, batch in enumerate(train_dataset):
    train_step(batch, model, optimizer, loss_fn, train_metrics)
    
    if (idx + 1) % logging_interval == 0:
        logging_metric(idx + 1, train_metrics)
    
    if (idx + 1) % valid_interval == 0:
        for batch in valid_dataset:
            valid_step(batch, model, loss_fn, valid_metrics)
        print("====Validation====")
        logging_metric(idx + 1, valid_metrics)
        print("==================")
        ckpt_manager.save()
```

위 코드를 이용하면 아래와 같이 학습이 진행된다. loss는 정상적으로 잘 떨어지고, 3가지 메트릭(Accuracy, Precision, Recall)도 함께 향상되는 모습을 볼 수 있다.

```
Step: 10|accuracy: 0.5148|precision: 0.5000|recall: 0.0628|loss: 0.6915
Step: 20|accuracy: 0.5781|precision: 0.6710|recall: 0.3854|loss: 0.6857
Step: 30|accuracy: 0.6961|precision: 0.6760|recall: 0.7078|loss: 0.6433
...
Step: 1080|accuracy: 0.8742|precision: 0.8844|recall: 0.8483|loss: 0.3087
Step: 1090|accuracy: 0.8492|precision: 0.8455|recall: 0.8358|loss: 0.3351
Step: 1100|accuracy: 0.8578|precision: 0.8795|recall: 0.8202|loss: 0.3345
====Validation====
Step: 1100|accuracy: 0.8564|precision: 0.8745|recall: 0.8344|loss: 0.3334
==================
```

## 3.4. 학습된 모델 Export

서빙을 위해 Tensorflow-Serving, TFLite 등에서 이용할 수 있도록 학습된 모델을 `SavedModel`형식으로 Export 한다. [Tensorflow SavedModel 튜토리얼](https://www.tensorflow.org/guide/saved_model?hl=ko)를 참고하여, 예측용 함수의 `input_signature`를 지정하고 이를 Export할 수 있다. 모델 자체에 토크나이징이 포함되어 있기 때문에, 입력으로 `tf.string`을 받는다. 또한 0,1 각각 클래스에 대한 확률 값을 출력으로 얻기 위해 모델 출력 값에 `softmax`연산을 진행한다.

```python
@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
def predict_fn(inputs):
    model_output = model(inputs)
    return tf.nn.softmax(model_output, -1)
print(predict_fn(tf.constant(["진짜 제 인생영화 ㅠㅠ"])))

signatures = {
    'serving_default': predict_fn.get_concrete_function(),
}
tf.saved_model.save(model, "./output/saved_model/00/", signatures)
```

위 실행의 결과로 아래 값을 얻을 수 있으며, 99%의 확률로 긍정(1)을 예측하는 것을 볼 수 있다. (모델이 잘 학습되었다!) 또한 string을 입력으로 하여 바로 모델 결과를 얻을 수 있다!

```
<tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[0.00181852, 0.99818146]], dtype=float32)>
```

위에서 Export한 모델과 tensorflow-servining을 이용하여 간단하게 모델 서버를 띄울 수 있다.

```shell
docker run -p 8501:8501 \
    -v /Users/baek-yeongmin/Documents/GitHub/tf-text-practice/output/saved_model/:/models/saved_model \
    -e MODEL_NAME=saved_model tensorflow/serving
```

위와 같은 입력을 보내고 결과 값을 확인해보면 같은 값을 반환하고, 띄워진 서버는 String 입력을 잘 처리함을 볼 수 있다.

```shell
curl -i -d '{"instances": ["진짜 제 인생영화 ㅠㅠ"]}' \
    -X POST http://localhost:8501/v1/models/saved_model:predict
```

```
HTTP/1.1 200 OK
Content-Type: application/json
Date: Sun, 11 Oct 2020 06:20:26 GMT
Content-Length: 58

{
    "predictions": [[0.00181851524, 0.998181462]
    ]
}
```

<br>

# 4. 후기

모델 코드 혹은 예측 함수에 토크나이징 과정을 포함하여 Tensorflow Graph를 구성한 후 이를 Export하면 텍스트를 입력으로 하는 모델 서버를 구성할 수 있다. 이를 이용하면 NLP 모델을 서빙하는 과정이 별도의 토크나이징 서버를 구성했던 이전에 비해 훨씬 간단해진다! `tensorflow-text`를 써보면서 앞으로 서빙에 있어서 `tensorflow`는 부동의 첫번째 선택지가 될 것 같다는 생각이 들었다. `tensorflow2`로 버전이 올라오면서 모델/학습 코드의 작성도 간편해졌는데, 연구 목적이 아니라 서빙까지 고려한다면 `torch`보다는 `tensorflow`를 선택하지 않을까..?
