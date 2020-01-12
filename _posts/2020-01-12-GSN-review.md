---
title: "GSN Review"
layout: post
categories:
  - paper-review
tags:
  - paper
  - nlp
  - graph
  - dialogue
last_modified_at: 2020-01-08T20:53:50-05:00
author: yeongmin
comments: true
---

이전 글에서 [Masked Hierarchical Transformer](https://baekyeongmin.github.io/paper-review/masked-hierarchical-transformer-review/)에 대해 리뷰했었는데요, 해당 논문의 레퍼런스들을 살펴보다가 대화를 Graph로 모델링하려는 또 다른 논문을 발견했습니다. 이번 포스트에서는 대화 그래프와 화자 그래프를 이용하여 response generation 문제를 풀고자 했던 [GSN: A Graph-Structured Network for Multi-Party Dialogues (IJCAI 2019)](https://arxiv.org/pdf/1905.13637.pdf)을 리뷰하려고 합니다.

# Main Idea

기존의 대화 모델링 방법들(HRED 등)은 대화를 발화들의 Sequential하게 구성되었다고 가정하고 모델링을 진행합니다. 하지만 Multi-Party Dialogue(단톡방과 같은 형태)에서는 화자가 여러명이기 때문에 여러 대화가 병렬적으로 동시에 상호작용할 수 있고, 이 가정이 유효하지 않습니다. 본 논문은 이러한 대화 구조를 모델링 하기 위해 대화 및 화자 정보를 그래프의 형태로 모델링합니다.

아래 그림과 같은 경우에, `utterance3`, `utterance4` 는 `utterance2`의 대답이지만, Sequential한 구조의 경우 이를 고려할 수 없습니다. 본 논문에서는 오른쪽의 Graph 구조를 고려한 모델링 방법을 제시합니다.

![dialogue](/images/GSN/dialogue.png){: width="80%"}{: .center}

# HRED

본 논문에서 베이스라인 모델로 제시하고있는 [Hierarchical Reccurent Encoder-Decoder(HRED)](https://arxiv.org/abs/1507.04808)는 Multi-turn의 대화를 모델링하기 위한 구조입니다. 크게 아래 그림과 같이 두개의 Reccurent 인코더 구조를 가지며, 각 발화를 인코딩하는 인코더, 해당 발화의 state를 받아서 세션 전체를 인코딩하는 인코더로 구성됩니다. 이렇게 컨텍스트를 인코딩하고, 이를 바탕으로 다음 발화를 생성합니다. 이 방법론은 논문의 서론에서 제시하듯이 각 발화들이 Sequential하게 구성되었다는 가정을 갖고 있습니다.

![hred](/images/GSN/HRED.png){: width="100%"}{: .center}

# Problem Formulation

본 논문에서는 "response generation" 문제를 풉니다. 이는 주어진 대화(컨텍스트)를 바탕으로 이에 알맞는 대답을 생성해내는 문제입니다. 저자들은 컨텍스트를 방향성 그래프, $$G(V, E)$$로 표현합니다. 여기서 $$V$$는 그래프를 구성하는 m개의 Vertice로, 대화를 구성하는 각 발화를 의미합니다. $$E$$ 는 그래프를 구성하는 edge로, $$j$$ 발화가 $$i$$ 발화의 대답이라면 $$e_{i,j} = 1$$ 나머지는 $$e_{i,j} = 0$$ 의 값을 가집니다. 각 발화($$V_i$$)는 $$s_i$$ 라는 발화를 대표하는 인코딩 벡터(발화 RNN에 의해 생성되는)로 나타내어집니다.

위와 같은 그래프 $$G(V, E)$$ 가 주어졌을 때, 가장 좋은 답변 $$\bar{r}$$를 찾는(생성하는) 문제를 풉니다.

$$\bar{r} = \underset{r}{\operatorname{argmax}} \log{P(r \vert G)} = \underset{r}{\operatorname{argmax}} \sum\limits_{i=1}^{\vert r \vert} \log{P(r_i \vert G, r_{<i})}$$

여기에 추가적으로 화자에 대한 정보 $$U$$(i,j가 같은 화자이면 $$U_{i,j} = 1$$ 나머지는 $$U_{i,j} = 0$$)를 모델링에 이용합니다.

# Graph-Structured Neural Network(GSN)

![architecture](/images/GSN/architecture.png){: width="100%"}{: .center}

본 논문에서 제시하는 모델의 구조는 위의 그림처럼 도식화될 수 있습니다. 모델은 크게 word-level encoder(`W-E`), utterance-level graph-structured encoder(`UG-E`), decoder(`UG-D`)의 3가지 부분으로 구성됩니다. `UG-E`이 본 논문에서 핵심으로 제시하는 구조로, Graph 구조를 이용하여 발화들의 관계를 인코딩하는 역할을 합니다.

## 1. Word-level Encoder(W-E)

세션에서 주어진 발화들에 대해 각각의 발화들을 하나의 representation으로 인코딩합니다. Bi-LSTM을 이용했고, 각 방향의 마지막 hidden state를 concat하여 세션에 속하는 각 발화들의 representation $$S = \{s_i, i\in \{i,...,m\}\}$$(i는 각 발화)를 얻습니다.

## 2. Utterance-level Graph-Structured Encoder(UG-E)

### 2.1. UG-E & Information Flow Over Graph

Graph 구조의 대화를 모델링하기 위해 dynamic iteration을 가지는 RNN 구조를 제시합니다. 한번의 iteration에서는 주어진 그래프상 각 선행 노드들의 정보만 다음 노드로 전달됩니다. 즉 $$E$$에서 $$e_{i,j} = 1$$인 경우, $$i$$ 의 정보를 이용하여 $$j$$ 의 정보를 업데이트 하고 이를 다음 iteration에서 이용합니다. [Main Idea](#main-idea)의 예시에서 2번의 iteration을 수행하면 아래 그림과 같이, `1번 발화`의 정보가 `3번 발화` 에 전달될 수 있습니다.

![info_flow](/images/GSN/info_flow.png){: width="80%"}{: .center}

word-level 인코딩의 결과 $$S = (s_1, s_2, s_3, s_4)$$ 가 주어졌을 때, $$l$$ 번째 iteration을 수식으로 나타내면 다음과 같습니다. 

$$s_i^l = s_i^{l - 1} + \eta \cdot \Delta s_{I \vert i} ^ {l - 1}$$

$$\Delta s_{I \vert i} ^ {l - 1} = \sum\limits_{i'\in \varphi} \Delta s_{I \vert i'} ^ {l - 1}$$

이를 해석해보면 $$l$$번째 iteration에서 발화의 representation은 이전 iteration($$l-1$$번째)의 representation에서 선행 노드로 부터의 정보(첫번 째 식 우변의 두번째 항)를 더하여 만들어집니다. 이 때 선행 노드로 부터의 정보는 두번 째 식과 같이 (선행 노드가 두 개 이상이라면) 모든 선행 노드들 정보의 합을 의미합니다. 위 식에서 $$\varphi$$ 은 새로운 정보를 얼마나 반영할지의 계수로, non-linear "squasing 함수"를 이용하여 다음과 같이 계산합니다.

$$\varphi = SQH(\Delta s_{I \vert i} ^ {l - 1}) = \frac{\alpha + \Vert \Delta s_{I \vert i} ^ {l - 1} \Vert}{1 + \Vert \Delta s_{I \vert i} ^ {l - 1} \Vert}$$

$$\alpha > 0$$는 하이퍼 파라메터로 위 식은 $$\Vert \Delta s_{I \vert i} ^ {l - 1} \Vert$$ 텀이 작으면 $$\alpha$$ 의 값을, 크면 1의 값을 갖게 됩니다. (정보의 양이 많을 수록 많이 반영됩니다.)

$$\Delta s_{I \vert i'} ^ {l - 1}$$ 텀(위 그림의 $$\otimes$$ 연산)은 선행노드($$i'$$)의 정보를 반영하여 현재 노드($$i$$)로 업데이트할 정보입니다. hidden state를 선행 노드의 representation $$s^{l-1}_{i'}$$, input state를 현재 노드의 representation $$s^{l-1}_{i}$$로 하는 GRU 모듈을 이용해 계산할 수 있습니다.

### 2.2. Bi-directional Information Flow

위의 방식대로 연산을 하면, 아래 그림 왼쪽의 forward flow와 같이 `utterance 3`, `utterance 4` 는 각각에 대한 대답을 생성하기 위해 `utterance 2` 의 정보를 이용할 수 있습니다. 그러나 `utterance 4` 의 정보는 `utterance 3` 의 정보를 이용할 수 없습니다.(반대의 경우도 동일) 즉 forward flow를 통해 선행 노드의 정보를 전달해 그래프 구조에 대한 정보를 모델에게 줄 수 있었지만, 형재 노드들(그래프의 depth가 같은 노드들)은 서로에 대한 정보를 이용할 수는 없게 되었고, 이는 생성 결과에 영향을 미칠 수 있습니다.

이를 해결하기 위해 저자들은 backward 방향의 정보도 이용하도록 하는 BIF(Bi-directional Information Flow) 알고리즘을 소개합니다. 아래 그림의 오랜지색 path로, forward path로 정보를 업데이트 하기 전에 backward path로 먼저 정보를 업데이트 합니다. 이렇게 되면 형제 노드들의 정보가 선행 노드를 거쳐 또 다른 형제 노드로 전파될 수 있습니다.

![bi_flow](/images/GSN/bi_flow.png){: width="80%"}{: .center}

### 2.3. Speaker Information Flow

대화에서 latent embedding space 상에서 화자에 대한 정보를 나타내는 것(임베딩 단에서 화자에 대한 정보를 추가 해주는 방식)은 대화 모델링에서 향상을 가져왔습니다. 하지만 이 방식은 다양하게 변경되는 화자와 그에 따른 생각 변화를 모델링하기는 부족합니다.

저자들은 동일 화자들의 발화를 그래프의 형태로 만드는 방법을 제시합니다. 또한 이렇게 만들어진 그래프를 이용하여 위의 [2.1번](#21-ug-e--information-flow-over-graph) 과 동일한 연산을 진행합니다. (단 모든 파라메터는 별도로 생성합니다.)

speaker information flow까지 합치면 $$l$$ 번째 representation의 최종 형태는 다음과 같습니다.(우변 세번째 항이 화자에 관련된 항입니다.)

$$s_i^l = s_i^{l - 1} + \eta \cdot \Delta s_{I \vert i} ^ {l - 1} + \lambda \cdot \Delta s_{I \vert i} ^ {\prime l - 1}$$

## 3. Decoder

디코더는 컨텍스트 인코딩 결과를 이용해서 진행되는데, 생성되기 전 마지막 발화의 Context 인코딩 결과와 GRU를 이용하여 연산을 진행합니다. 독특한 점은 인코더에서 마지막 발화의 $$l$$번의 iteration 중간 값을 갖고 있다가, 이와 디코더 현 시점의 hidden state 사이의 attention을 계산하여 이를 다음 스탭의 디코더 계산에 이용했다는 점입니다. (번역에서 각 단어의 encoder 결과를 이용하여 attention을 계산하는 과정과 유사함.)

# Experiment

## 1. Setup

- 데이터는 Ubuntu Dialogue Corpus를 이용했습니다.
  - 코퍼스에 등장하는 "@" 심볼을 이용해서 그래프를 구성하였습니다. "A @ B" 의 뜻은 이 발화가 화자 A에 의해 화자 B에게 전달됨을 뜻합니다.
- 비교 대상으로 HRED와 Seq2Seq 모델 (+ 화자 정보 임베딩)을 이용하였습니다.
- 학습 파라메터
  - 단어 임베딩: 300차원
  - Word-level encoder & decoder: 2-layer
  - Optimizer: Adam (lr: 0.0001)
  - epoch: 25

## 2. Result

![result_1](/images/GSN/result_1.png){: width="100%"}{: .center}

위와 같이 제시한 방법론은 기존 방법들에서 화자 정보를 추가한 경우들을 포함하여 모든 베이스라인을 뛰어넘는 성능을 보여주었습니다. iteration의 횟수가 증가할 수록 multi-hop의 노드들을 반영할 수 있고, 더 좋은 결과를 얻음을 알 수 있습니다. 또한 제시한 방법은 화자의 정보가 없는 경우에도 (화자 정보를 사용한) 모든 베이스 라인 성능을 능가했습니다. 표에는 등장하지 않지만 4,5번째 iteration 까지 실험한 경우 성능이 3 iteration과 유사하고, 6번째 iteration 부터는 성능이 떨어진다고 언급했습니다. (이는 GCN등 다른 Graph 기반 모델들과 비슷한 양상입니다.)

저자들은 제시한 방법론이 Multi-party인 경우 외에 Sequential 한 경우에도 일반적으로 잘 동작함을 보이기 위해 Sequential 데이터 (화자 두 명만으로 구성)와 Graph 데이터(Multi-party 데이터)를 구분하여 추가적인 실험을 진행했습니다.

![result_2](/images/GSN/result_2.png){: width="100%"}{: .center}

결과적으로 두 경우 모두 베이스라인을 뛰어넘었고, 제시한 대화의 구조 그래프와 화자 정보 그래프를 이용한 컨텍스트 인코딩 방식(UG-E)이 일반적인 대화 모델링에 있어서 기존 방법들을 뛰어넘었음을 증명했습니다.

# Reference

[Iulian V Serban, Alessandro Sordoni, Yoshua Bengio, Aaron Courville, 2016 and Joelle Pineau. Building end-to-end dialogue systems using generative hierarchical neural network models. In Thirtieth AAAI Conference on Artificial Intelligence, 2016.](https://arxiv.org/abs/1507.04808)

[Hu, W.; Chan, Z.; Liu, B.; Zhao, D.; Ma, J.; and Yan, R.
2019 Gsn: A graph-structured network for multi-party dialogues. (IJCAI 2019)](https://arxiv.org/abs/1911.10666)