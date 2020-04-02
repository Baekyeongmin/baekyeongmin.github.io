---
title: "Trnasformer-XL Review"
layout: post
categories:
  - paper-review
tags:
  - paper
  - nlp
last_modified_at: 2020-04-01T20:53:50-05:00
author: yeongmin
comments: true
---


이번 글에서는 ACL 2019에서 발표된 ["Trnasformer-XL: Attentive Language Models Beyond a Fixed-Length Context"](https://arxiv.org/abs/1901.02860)를 리뷰하려고 합니다. 본 논문은 기존의 Transformer 구조를 이용한 고정된 길이(Fixed-Length) Language Model의 한계점을 지적하고 더 긴 Long-term dependancy를 이용할 수 있는 새로운 방법을 제시합니다. 또한 다양한 NLU 테스크들에서 SOTA성능을 보이고 있는 [XLNet](https://arxiv.org/abs/1906.08237)과 동일한 저자들이 작성하였고, Transformer-XL의 많은 부분을 XLNet에서 이용하고 있습니다. 이번 포스트에서는 논문과 함께 [저자들의 구현체](https://github.com/kimiyoung/transformer-xl/tree/master/pytorch)도 함께 살펴봅니다.

# 1. Main Idea
기존의 Transformer기반의 LM(vanilla transformer model)은 코퍼스를 여러 개의 segment들로 나누고, 아래 림과 같이 각 segment별로 *"해당 segment내에서"* Langauge Modeling의 Auto-regressive한 Objective를 학습했습니다. 따라서 segment의 고정된 최대 길이 내에서만 학습이 이루어지므로, 해당 범위를 벗어나는 long-term dependancy는 학습할 수 없습니다. 또한 각 segment는 문장 등의 의미있는 단위로 나눠진 것이 아닌 단순하게 연속적인 symbol들(token, word 등)의 조각들로 구성되기 때문에 해당 segment의 처음 몇개의 symbol들을 예측하기에는 필요한 정보의 양이 부족한 *context fragmentation* 문제가 발생합니다. 저자들은 이러한 vanilla Transformer가 갖고 있는 문제들을 해결하기 위해 Trnasformer-XL(extra long)이라는 방법을 제시합니다.

# 2. Transformer

![transformer](/images/Transformer-XL/transformer.png){: width="100%"}{: .center}

Transformer는 ["attention is all you need"(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)에서 소개되었고, 재목 그대로 "Attention"만을 이용하여 시퀀스를 모델링합니다. 이 방법은 특정 시퀀스 내의 모든 토큰들 사이의 Attention(Self-Attention)과 이에 따른 가중합을 통해 시퀀스를 모델링하는데, 이로 인해 RNN기반의 모델들이 갖고 있던 1)long-term dependancy 문제, 2) optimizing 문제(Vanishing/Exploding gradient)를 해결합니다.

["Character-level language modeling with deeper self-attention" Al-Rfou et al. 2018)](https://arxiv.org/abs/1808.04444?source=post_page---------------------------)에서는 Transformer구조로 character단위의 Language Modeling 문제를 풀었는데, 기존의 RNN기반의 모델들을 큰 격차로 능가하는 성능을 보여주었습니다. 하지만 이 방법은 다음과 같은 문제를 갖습니다.
1. 고정된 길이의 segment를 입력으로 이용하기 때문에, 해당 길이 이상의 의존성을 학습하기 힘들다.
2. 각 segment는 의미적 경계(문장 등)을 고려하지 않고 단순히 연속적인 token들을 잘라서 이용했기 때문에, segment의 시작 몇 토큰을 예측하기에는 정보량이 부족하다. - *context fragmentation*문제

# 3. Trnasformer XL

위에서 제시한 문제점을 해결하기 위해, 저자들은 Transformer 구조의 reccurence한 형태를 제시합니다. 이는 연속된 segment를 모델링 할 때, 각 segment를 독립적으로 모델링(기존의 방식)하는 것이 아니라 특정 segment의 모델링에 이전 segment의 정보(각 layer의 hidden state)를 이용하는 방법입니다. 이에 따라 여러 segment사이의 의존성도 파악할 수 있게 되어 고정된 길이의 의존성 문제를 해결하게 되고, context fragment 문제 또한 해결할 수 있게 됩니다.

본 모델은 특정 시점($$t$$) 이전의 토큰들($$x_{<t}$$)이 주어졌을 때, $$t$$ 시점에 등장할 토큰을 예측하는 Language Modeling 문제를 풉니다. 이는 $$P(x) = \prod_tP(x_t \mid x_{<t})$$와 같이 Auto Regressive한 방식으로 나타낼 수 있고, $$P(x)$$의 분포를 추정합니다.
일반적으로 Neural Network를 이용한 방법들은 다음과 같습니다.
1. $$t$$시점 이전의 정보들을 고정된 크기의 벡터로 만듭니다.
2. 이 벡터와 word embedding을 곱하여 logit을 만듭니다.
3. softmax함수를 이용하여 logit을 다음 단어에 대한 확률 분포로 만듭니다.

## 3.1 Vanilla Transformer Language Models

![vanilla-Transformer](/images/Transformer-XL/vanilla.jpg){: width="100%"}{: .center}

Transformer로 매우 긴 컨텍스트를 모델링 하려면 어떻게 해야할까요? 가장 간단한 방법은 모든 컨텍스트를 모델의 입력으로 제공하는 것인데, 이 방법은 메모리/계산상의 한계로 불가능하다고 볼 수 있습니다. 따라서 기존의 방법론들은 이를 코퍼스를 여러개의 짧은 segment로 나누고 각 segment를 독립적으로 모델링하도록 학습하는 방식으로 근사합니다.(*vanilla model*, 위 그림의 왼쪽) 이 방법은 transformer의 장점(Optimizing/long-term dependency)를 완전히 이용하고 있다고 볼 수 없습니다.

이 모델의 평가를 진핼할 때, 위 그림의 오른쪽과 같이 학습과 동일한 컨텍스트 길이로 한 칸씩 옮겨가며 다음 토큰을 예측하는 방식으로 진행합니다.(sliding window) 이 방식은 각 단계에서 학습 때 주어진 최대 길이의 정보를 이용할 수 있지만, 많은 계산량을 요구하고 느린 속도를 갖습니다.

## 3.2 Segment-Level Reccurence with State Reuse

![transformer-xl](/images/Transformer-XL/transformer_xl.png){: width="100%"}{: .center}

Vanilla Transformer의 문제를 해결하기 위해 Transformer 구조의 recurrence 방법을 적용합니다. 학습이 진행되는 동안, 각 segment의 연산 결과들을 다음 segment가 이용할 수 있도록 저장(fixed/cached)합니다. 현재 segment에서는 모델링을 직전 segment의 정보를 이용할 수 있습니다. 즉 위 그림의 왼쪽 부분과 같이 하나의 segment를 모델링 하기 위해 두 개의 연속된 segment의 정보를 이용합니다.

$$\tilde{h}^{n-1}_{\tau+1} = [SG(h^{n - 1}_{\tau + 1}) ; h^{n-1}_{\tau + 1}]$$ 
  
$$\tau - 1$$과 $$\tau$$ 시점의 $$n-1$$ 번째 layer의 hidden state를 concat하여 Transformer $$n$$ 번째 layer에서 이용될 입력을 구성합니다. SG는 Stop Gradient로, Backpropagation을 할 때, 이전 segment의 hidden state를 만들기 위해 이용되었던 prameter는 학습되지 않습니다.(Grandient가 전파되지 않습니다.)

$$q^n_{\tau + 1}, k^n_{\tau + 1}, v^n_{\tau + 1} = h^{n - 1}_{\tau + 1}W^T_q, \tilde{h}^{n-1}_{\tau+1}W^T_k, \tilde{h}^{n-1}_{\tau+1}W^T_v$$

첫 번째 식에서 만들어진 hidden state를 이용하여 Self attention에 이용될 Key, Query, Value 벡터를 만듭니다. 이 때, 현재 segment에 대한 hidden state를 계산하면 되기 때문에 Query는 $$h^{n - 1}_{\tau + 1}$$를 이용하여 계산됩니다.

$$h^n_{\tau + 1} = Transformer\_Layer(q^n_{\tau + 1}, k^n_{\tau + 1}, v^n_{\tau + 1})$$

만들어진 Query, Key, Value를 이용하여 현재 Segment에 대한 현재 Layer의 hidden state($$h^n_{\tau + 1}$$)를 만듭니다.

이와 같이 연속된 두 segment 사이에 recurrence 매커니즘이 **segment-level**로 적용되었습니다. 결과적으로 $$t$$ 시점의 segment는 $$<t$$ 시점의 정보들 또한 포함할 수 있습니다. 하지만 $$h^n_{\tau+1}$$은 $$h^{n-1}_{\tau}$$를 이용하는 의존관계 때문에 segment 하나 당 하나의 layer씩 밀리게 됩니다. (동일한 layer에서 recurrence연결이 있는 RNN과 차이가 있습니다.) 따라서 최대 의존관계는 segment length $$\times$$ layer 수로 제한됩니다. 위 그림의 오른쪽의 경우에는 $$4(segment \space length) \times 3 (layer)$$로 최대 12길이의 의존성을 가질 수 있습니다.

평가시 속도 측면에서 보면, 이전 segment의 state를 저장함으로써, sliding window 방식(vanilla transformer)을 이용하지 않아도 되기 때문에, 기존의 방법보다 약 +1800배 빠르게 연산을 진행할 수 있습니다. 또한 평가 환경에서 장치의 메모리 등이 허용하는 한에서 이전 segment의 길이를 더 길게 가져감으로써, 더 긴 의존성을 만들 수 있습니다. 이 때 이전 segment는 memory augmented neural network의 메모리와 유사한 역할을 수행할 수 있습니다.

## 3.3 Relative Positional Encodings

앞에서 제시한 구조를 실현하기 위해서는 한가지 문제를 더 해결해야하는데, Positional Encoding 입니다. 기존의 Transformer는 특정 위치의 토큰의 embedding을 계산할 때, 해당 토큰의 Word Embedding과 토큰 위치의 Positional Encoding값을 더합니다. 따라서 각 위치마다 고유한 embedding 값을 갖습니다. 이를 recurrence 매커니즘에 적용해보면, 다음과 같습니다.

$$h_{\tau + 1} = f(h_{\tau}, E_{s_{\tau + 1} + U_{1:L}})$$

$$h_{\tau} = f(h_{\tau - 1}, E_{s_{\tau} + U_{1:L}})$$

$$E$$ 는 word embedding $$U$$ 는 Positional encoding입니다. 이 때 이전 state를 만들 때와 현재 state를 만들 때 사용한 positional encoding($$U$$)이 동일하여, 모델이 $$x_{\tau, j}$$ 와 $$x_{\tau + 1, j}$$ 를 구분할 수 없습니다.

positional encoding은 모델에게 토큰의 위치에 대한 단서/bias(어디에 attend 해야 하는지)를 제공합니다. 동일한 목적을 위해서는 위치 정보를 일반적인 transformer의 방식과 같이 초기 임베딩에 포함시키는 것 대신 각 layer의 attention score에 직접 포함시킬 수 있습니다. 또한 attention에서는 각 토큰의 query, key 벡터 사이의 유사도를 계산하는데, 이 때 각 토큰의 절대적 위치를 아는 것 보다는 두 토큰 사이의 상대적인 거리를 아는 것이 더 중요합니다. 즉 시간적 정보를 "절대적"(기존의 방식)이 아닌 "상대적"으로 정의하는 것이 조금 더 직관적이고, 일반화 가능합니다.

이러한 근거에 따라 저자들은 기존 Transformer의 방식에서 부터 새로운 Relative Positional Encoding 방식을 제안합니다. 

$$A_{i, j}^{abs} = (E_{x_i}^T + U_i)$$

Relative Postion을 나타내는 $$R \in \mathbb{R}^{L_{max} \times d}$$ ($$L$$은 segment 최대 길이, $$d$$는 hidden size)를 만듭니다. 이 메트릭스의 $$i$$번째 행은 두 토큰 사이의 상대적 거리가 $$i$$인 position 정보를 나타냅니다.