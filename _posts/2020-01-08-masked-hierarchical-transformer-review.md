---
title: "Masked Hierarchical Transformer Review"
layout: post
categories:
  - paper-review
tags:
  - Dialogue-modeling-review
last_modified_at: 2020-01-08T20:53:50-05:00
author: yeongmin
comments: true
---

대화는 문어체(위키피디아, 책 등)와 많은 차이점을 갖고 있습니다. 자주 사용되는 어휘, 어투 등 각 발화의 형태적인 차이뿐만 아니라, 두 명 이상의 화자가 서로 상호작용을 함으로써 생기는 구조에서도 큰 차이가 있습니다. 이전 포스트들에서 살펴봤듯이, Transformer 구조 기반의 Self-supervised learning(BERT) 방식의 학습법이 대부분 NLP 테스크들의 최고 성능(State-of-the-art)를 달성했습니다. 하지만 이러한 방식들은 주로 문어체 데이터로 pre-training이 이루어 졌기 때문에 대화에 바로 적용하기 힘듭니다. 이를 위해 다양한 방법이 제시되었는데요, 본 포스트에서는 pre-trained BERT를 이용하여 대화의 구조를 모델링 하고자 한 ["Who Did They Respond to?
Conversation Structure Modeling Using Masked Hierarchical Transformer"(AAAI 2020)](https://arxiv.org/abs/1911.10666) 를 리뷰하려고 합니다.

<br>

# Main Idea

- Google의 pre-trained BERT를 문장 인코더로 이용하고, 이 위에 문장의 구조를 파악할 수 있는 추가적인 Transformer 인코더를 학습시킴으로써, 대화 구조를 모델링하고자 했습니다. 즉, 문장 인코딩의 결과를 이용하여 컨텍스트를 인코딩하는 2단계의 계층적인 구조를 제시합니다.
- 컨텍스트 인코더를 학습할 때, 대화에서 각 발화의 의존관계를 파악할 수 있도록 하는 Masking 방법을 제시합니다.

<br>

# Conversation Disentanglement

본 논문에서 대화 구조 모델링을 위해 푼 문제입니다. 대화는 여러 개의 "thread" 로 구성되어 있어서 분리할 수 있기 때문에, 이를 분리하는 문제를 풉니다. 구체적으로, 아래 그림과 같이 여러 개의 발화로 구성된 대화에서 "특정 발화"가 "이전의 대화 히스토리"에서 어떤 발화의 대답인지 "Reply-to"관계를 파악하는 것이 목적입니다. 이를 통해 주어진 대화 구조를 계층적인 그래프의 형태로 분리할 수 있습니다.

![problem](/images/HMT/problem.png){: width="100%"}{: .center}

본 논문 이전에는 이를 문장쌍의 문제로 접근했습니다. 즉 문장쌍 `(문장1, 문장2)` 가 주어졌을 때, `문장1` 의 대답이 `문장2` 인지 구분하는 이진 분류 문제를 풀었습니다. 따라서 대화의 `문장1` 이외의 다른 대화 히스토리는 고려하지 않았습니다. 이 방식으로 위의 예제의 문제를 푼다면 `utterance4`는 `utterance2`, `utterance3` 과 모두 reply-to 관계를 형성할 수 있습니다. 하지만 모든 대화의 히스토리를 고려한다면, `utterance2` 만 답이라는 것을 알 수 있습니다. 본 논문에서는 이를 해결하기 위해 모든 히스토리를 고려하는 방식을 제시합니다.

<br>

# Masked Hierarchical Transformer Model

모델은 아래 그림과 같이 크게 두 개의 인코더로 구성됩니다. 입력으로 발화들의 시퀀스 $$S_1, S_2, ..., S_L$$ 를 받는데, 이를 이용하여 마지막 발화 ($$S_L$$)가 대화 히스토리($$S_1, ... S_{L-1}$$) 중 어떤 발화와 reply-to 관계가 있는지를 풉니다. 모델 학습 과정은 다음과 같습니다.

![model](/images/HMT/model.png){: width="80%"}{: .center}

1. 입력으로 들어온 발화들의 시퀀스($$S_1, S_2, ..., S_L$$)를 각각 발화 인코더(위 그림의 Shared Utterance Encoder)로 인코딩합니다. 각 발화는 BERT의 입력과 동일하게 `[CLS]` + `발화` + `[SEP]`과 같은 형태로 구성됩니다. 이 인코더는 모든 발화들에 대해 공유되며, `"pre-trained BERT Base, Uncased"`의 가중치로 초기화됩니다. 각 발화별 인코딩 결과의 `[CLS]`의 representation(pooled_output)을 해당 발화의 대표 특징 벡터, 대표 representation($$V_1, V_2, ... V_L$$)으로 이용합니다. 즉, 각 발화마다 하나의 벡터를 만듭니다.
2. 각 발화들의 특징 벡터($$V_1, V_2, ... V_L$$)를 또 다른 transformer 인코더(위 그림의 Masked Transformer)를 이용하여 인코딩합니다. 각 발화의 특징 벡터들 사이의 self-attention을 통해 발화들 사이의 관계를 모델링합니다. 이 때, 더 효과적인 모델링을 위해 특별한 masking 전략을 사용합니다. (이는 뒤의 세션에서 자세히 설명합니다.) 이를 통해 각 발화별로 ($$\tilde{V_1}, \tilde{V_2}, ... \tilde{V_L}$$)를 만듭니다.
3. 히스토리의 각 발화별 최종 representation($$\tilde{V_1}, \tilde{V_2}, ... \tilde{V_{L-1}}$$)와 추가적인 Fully-connected layer를 이용하여, 최종적으로 티겟 발화($$S_L$$)가 어떤 발화와 reply-to 관계가 있는지 파악합니다. 이 때, 다음과 같이 데이터셋 별로 다른 학습 전략을 사용합니다.
    - Reddit dataset: 하나의 발화는 하나의 부모 발화를 가지기 때문에 특정 타겟 발화에 따라 하나의 정답만 가집니다. 따라서, 각 히스토리 발화들의 representation 마다 Fully-connected layer를 통해 하나의 logit 값($$t_i$$)을 얻고, 모든 히스토리 발화들에 대해 softmax 연산을 하여, 정답 발화(타겟 발화의 부모 발화)의 확률이 가장 높게 나오도록 학습합니다. 즉, 발화들 사이의 상대적인 우위를 학습하는 랭킹 방식입니다.
    
    $$t_i = W_o \tilde{V_i} + b_o, i=1,...L-1, W_o \in \mathbb{R}^{Hidden\_size \times 1}$$
    
    $$\tilde{Y_i} = \frac{exp(t_i)}{\sum_{j=1}^{L-1} exp(t_i)}$$

    $$loss = - \sum\limits_{i=1}^{L-1}y_i \log(\tilde{Y_i})$$

    - Ubuntu IRC: Reddit과 다르게 하나의 발화가 여러 개의 부모 발화를 가질 수 있고, 자기 자신이 부모 발화가 될 수 있습니다. 따라서, 각 히스토리 발화와 타켓 발화(자기 자신이 부모인 경우 때문에)에 대해 독립적으로 reply-to 관계인지 아닌지에 대한 binary classification 문제를 풉니다.

## 1. Ancestor Masking

self-attention layer에서 attend할 수 있는 대상은 attention mask($$ M = L \times L$$ 크기의 행렬) 에 의해 결정됩니다. 두 representation 사이에 mask가 0인 경우($$M_{ij}=0$$) attention이 생길 수 없고, 1인 경우($$M_{ij}=1$$) attention이 생길 수 있습니다. 일반적으로 generation model(GPT 등)에서 현제 시점 이후 토큰이나 Padding값에 attention이 가지않게 하기 위해 사용됩니다. 본 논문은 Masked Transformer(두 번째 Transformer)의 연산에서 다음과 같은 전략에 따라 Mask를 만들었습니다.

- 모든 히스토리 발화는 타겟 발화의 부모 발화 후보이기 때문에, 타겟 발화를 attend할 수 있습니다. 따라서 히스토리 발화들(부모 발화 후보들)은 타겟 발화를 고려하여 자신의 representation을 만들게 됩니다. ($$M_{iL} = 1$$)
- 모든 발화들은 자기 자신을 attend 할 수 있습니다. ($$M_{ii} = 1$$)
- 타겟 발화가 아닌 모든 발화들(히스토리 발화들)은 자신의 조상 발화(부모, 부모의 부모, ...)에 attend 할 수 있습니다. 즉, 히스토리 발화들의 대화 구조는 사전에 정해져있습니다.
- 위 조건 외에 나머지는 모두 attend할 수 없습니다.

![attention-mask](/images/HMT/attention_mask.png){: width="80%"}{: .center}

## 2. Two-Stage Training

발화 인코더는 pre-trained bert에 의해 초기화 되지만, Masked Transformer는 pre-training이 없습니다. 따라서 두 인코더를 함꼐 학습시킨다면 learning rate에 따라 다음과 같은 문제가 발생할 수 있습니다.
- learning rate이 큰 경우: 발화 인코더가 pre-training에서 배웠던 지식들을 잊는다.
- learning rate이 작은 경우: Masked Transformer의 학습이 잘 이루어지지 않는다.

위 문제를 해결하기 위해 본 논문에서는 다음과 같이 두 단계의 학습단계를 거칩니다.
1. 큰 learning rate으로 발화 인코더의 파라메터를 학습하지 않고, Masked Transformer의 파라메터만 학습합니다.
2. 상대적으로 작은 learning rate으로 두 인코더의 파라메터를 동시에 학습합니다.

# Experiments

## 1. Dataset
1. Raddit Small: Reddit으로 부터 만들어진 데이터로, 타이틀에 달린 댓글과 해당 댓글에 달린 댓글들로 구성됩니다.(이 관계를 통해 relpy-to관계를 형성합니다.) 몇 가지 과정(삭제된 댓글, 최대 댓글 개수)을 통해 데이터를 정제합니다. 상대적으로 관계 그래프가 깊지 않습니다.(주로 그래프가 펴져있는 형태입니다.)
2. Raddit Large: 위의 문제는 상대적으로 깊지 않은 그래프로 쉬운 문제이기 때문에, 저자들이 원본 Reddit dump로 더 어렵고(최소 깊이를 6으로 제한함) 많은 양의 데이터를 만들었습니다. 
3. Ubuntu IRC: 이 데이터는 Ubuntu 대화 데이터를 직접 레이블링 하여 만들어졌습니다. Reddit이 각 발화당 하나의 부모 발화를 갖는데 반해, 이 데이터는 하나의 발화에 여러 개의 부모 발화가 존재할 수 있습니다.

## 2. Experiment Setting

- 발화 인코더: `BERT-base` 와 동일한 설정을 이용했습니다.
    - Transformer Layer 수: 12
    - Multi-head 수: 12
    - Hidden layer size: 768
    - Intermediate layer size: 3072
- Masked Transformer: 발화 인코더에 비해 상대적으로 작은 크기의 설정을 이용했습니다.
    - Transformer Layer 수: 4
    - Multi-head 수: 4
    - Hidden layer size: 300
    - Intermediate layer size: 1024
- 학습 파라메터(Reddit):
    - Learning rate: 1e-4(first-stage), 1e-5(second-stage)
    - Optimizer: Adam
    - LR scheduler: BERT와 동일
    - Batch size: 32(first-stage), 8(second-stage)
    - Epoch: 10
    - Early stopping 사용
- 학습 파라메터(Ubuntu IRC):
    - Learning rate: 1e-5(first-stage), 1e-7(second-stage)
    - Batch size: 32(first-stage), 4(second-stage)
    - 나머지는 Raddit과 동일

## 3. Result

**Raddit Corpus** : 다음과 같은 두가지 Metric에 대해 평가를 진행했습니다.
- Graph Acc: 특정 발화의 부모 발화를 맞추는 정확도
- Conv Acc: 타이틀부터 시작하여 대화 전체의 구조를 reconstruction 했을 때, ground-truth와 일치 여부

![reddit-small](/images/HMT/reddit_small.png){: width="80%"}{: .center}

![reddit-large](/images/HMT/reddit_large.png){: width="80%"}{: .center}

위 그림과 같이 Raddit Small, Large에 대해 모든 baseline을 뛰어넘는 성능을 보여주었습니다. Large셋은 상대적으로 어렵기 때문에, 정확도가 매우 낮은 모습을 볼 수 있습니다.

**Ubuntu IRC** : 본 데이터셋에는 학습에 도움이되는 추가적인 피쳐(년도, 대화의 빈도, 메세지 타입, 시간 간격 등)가 제공되기 때문에 이를 사용한 경우와 사용하지 않은 경우를 분리했습니다.

대화의 graph에서 connected component를 cluster로 정의하고, 다음과 같은 Metric에 대해 평가를 진행했습니다.
- Variation of Information(VI): 데이터셋을 제시한 논문에서 사용한 metric
- One-to-One overlap(1-1): cluster의 1대 1 대응
- Precision(P), Recall(R), F1(F)

![ubuntu](/images/HMT/ubuntu_irc.png){: width="80%"}{: .center}

ubuntu 데이터에 대해서도 위의 그림과 같이 모든 baseline 성능을 뛰어넘는 결과를 보여주었습니다.

## 4. Ablation Study

본 논문에서 제시한 방법들의 효과를 검증하기 위해 raddit-small 데이터를 이용하여 ablation study를 진행했습니다.

### 4.1. Importance of Mask

제시한 Masking 전략의 효과를 검증하기 위해 모든 발화들에 attend할 수 있도록 mask를 없앤 경우(mask의 모든 값이 1, w/o mask)와 비교를 진행했고, 아래 그림과 같이 Masking 전략을 이용한 경우, 훨씬 높은 결과를 얻을 수 있었습니다. 이는 직관적으로, attention이 모든 발화로 분산되면 학습이 이려워지기 때문입니다. 또한 이 경우 BERT pairwise baseline보다 못한 성능으로, 적절한 masking없이 모든 발화를 모델에게 제공하는 것은 큰 혼동을 준다고 볼 수 있습니다.

![ablation-mask](/images/HMT/ablation_mask.png){: width="60%"}{: .center}

### 4.2. Importance of Ancestor Depth

본 논문에서는 특정 발화의 모든 조상 발화들(Ancestor, 부모의 부모, 부모의 부모의 부모 ...)을 모두 attend할 수 있도록 Mask를 구성합니다. 또다른 Masking 방식으로 d개의 조상 발화들만 attend 하도록 할 수도 있는데, d에 따라 성능이 어떻게 변하는지 실험을 진행했습니다. raddit-small에서 가장 깊은 깊이는 12이므로 d=1~12까지 실험을 했고, 결과는 아래 그림과 같이 모든 조상을 다 이용할 때가 가장 좋음을 볼 수 있습니다.

### 4.3. Temporal Mask

또 다른 Masking 방식으로, 대화 구조를 이용하지 않고 특정 발화 기준으로 이전 t개의 발화를 attend하도록 mask를 만들어 실험을 진행했습니다. 아래 그림과 같이 전체적으로 대화 구조를 이용한 경우보다 낮은 성능을 보였고, t가 증가할수록(이전의 더 많은 발화를 볼수록) 미세한 정확도 향상이 있습니다. 따라서 대화 구조에 대한 정보가 큰 영향을 가진다고 볼 수 있습니다.

![ablation-depth](/images/HMT/ablation_depth.png){: width="60%"}{: .center}

<br>

# Reference

- Henghui Zhu, Feng Nan, Zhiguo Wang, Ramesh Nallapati, Bing Xiang. Who Did They Respond to?
Conversation Structure Modeling Using Masked Hierarchical Transformer In Association for the Advancement of Artificial Intelligence(AAAI), 2020.
