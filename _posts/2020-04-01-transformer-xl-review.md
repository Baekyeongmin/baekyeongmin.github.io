---
title: "Transformer-XL Review"
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


이번 글에서는 ACL 2019에서 발표된 ["Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"](https://arxiv.org/abs/1901.02860)를 리뷰하려고 합니다. 본 논문은 기존의 Transformer 구조를 이용한 고정된 길이(Fixed-Length) Language Model의 한계점을 지적하고 더 긴 의존성을 이용할 수 있는 새로운 방법을 제시합니다. 또한 다양한 NLU 테스크들에서 SOTA성능을 보이고 있는 [XLNet](https://arxiv.org/abs/1906.08237)과 동일한 저자들이 작성하였고, Transformer-XL의 많은 부분을 XLNet에서 이용하고 있습니다.

# 1. Main Idea
기존의 Transformer기반의 LM(vanilla transformer model)은 코퍼스를 여러 개의 시그먼트들로 나누고, 아래 그림과 같이 각 시그먼트별로 *"해당 시그먼트내에서"* Langauge Modeling의 Auto-regressive한 Objective를 학습했습니다. 따라서 시그먼트의 고정된 최대 길이 내에서만 학습이 이루어지므로, 해당 범위를 벗어나는 long-term dependancy는 학습할 수 없습니다. 또한 각 시그먼트는 문장 등의 의미있는 단위로 나눠진 것이 아닌 단순하게 연속적인 symbol들(token, word 등)의 조각들로 구성되기 때문에 해당 시그먼트의 처음 몇개의 symbol들을 예측하기에는 필요한 정보의 양이 부족한 *context fragmentation* 문제가 발생합니다. 저자들은 이러한 vanilla Transformer가 갖고 있는 문제들을 해결하기 위해 Transformer-XL(extra long)이라는 방법을 제시합니다.

# 2. Transformer

![transformer](/images/Transformer-XL/transformer.png){: width="100%"}{: .center}

Transformer는 ["attention is all you need"(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)에서 소개되었고, 재목 그대로 "Attention"만을 이용하여 시퀀스를 모델링합니다. 이 방법은 특정 시퀀스 내의 모든 토큰들 사이의 Attention(Self-Attention)과 이에 따른 가중합을 통해 시퀀스를 모델링하는데, 이로 인해 RNN기반의 모델들이 갖고 있던 1)long-term dependancy 문제, 2) optimizing 문제(Vanishing/Exploding gradient)를 해결합니다.

["Character-level language modeling with deeper self-attention" Al-Rfou et al. 2018)](https://arxiv.org/abs/1808.04444?source=post_page---------------------------)에서는 Transformer구조로 character단위의 Language Modeling 문제를 풀었는데, 기존의 RNN기반의 모델들을 큰 격차로 능가하는 성능을 보여주었습니다. 하지만 이 방법은 다음과 같은 문제를 갖습니다.
1. 고정된 길이의 시그먼트를 입력으로 이용하기 때문에, 해당 길이 이상의 의존성을 학습하기 힘들다.
2. 각 시그먼트는 의미적 경계(문장 등)을 고려하지 않고 단순히 연속적인 token들을 잘라서 이용했기 때문에, 시그먼트의 시작 몇 토큰을 예측하기에는 정보량이 부족하다. - *context fragmentation*문제

# 3. Transformer XL

위에서 제시한 문제점을 해결하기 위해, 저자들은 Transformer 구조의 reccurence한 형태를 제시합니다. 이는 연속된 시그먼트를 모델링 할 때, 각 시그먼트를 독립적으로 모델링(기존의 방식)하는 것이 아니라 특정 시그먼트의 모델링에 이전 시그먼트의 정보(각 layer의 hidden state)를 이용하는 방법입니다. 이에 따라 여러 시그먼트사이의 의존성도 파악할 수 있게 되어 고정된 길이의 의존성 문제를 해결하게 되고, context fragment 문제 또한 해결할 수 있게 됩니다.

본 모델은 특정 시점($$t$$) 이전의 토큰들($$x_{<t}$$)이 주어졌을 때, $$t$$ 시점에 등장할 토큰을 예측하는 Language Modeling 문제를 풉니다. 이는 $$P(x) = \prod_tP(x_t \mid x_{<t})$$와 같이 Auto Regressive한 방식으로 나타낼 수 있고, $$P(x)$$의 분포를 추정합니다.
일반적으로 Neural Network를 이용한 방법들은 다음과 같습니다.
1. $$t$$시점 이전의 정보들을 고정된 크기의 벡터로 만듭니다.
2. 이 벡터와 word embedding을 곱하여 logit을 만듭니다.
3. softmax함수를 이용하여 logit을 다음 단어에 대한 확률 분포로 만듭니다.

## 3.1 Vanilla Transformer Language Models

![vanilla-Transformer](/images/Transformer-XL/vanilla.jpg){: width="100%"}{: .center}

Transformer로 매우 긴 컨텍스트를 모델링 하려면 어떻게 해야할까요? 가장 간단한 방법은 모든 컨텍스트를 모델의 입력으로 제공하는 것인데, 이 방법은 메모리/계산상의 한계로 불가능하다고 볼 수 있습니다. 따라서 기존의 방법론들은 이를 코퍼스를 여러개의 짧은 시그먼트로 나누고 각 시그먼트를 독립적으로 모델링하도록 학습하는 방식으로 근사합니다.(*vanilla model*, 위 그림의 왼쪽) 이 방법은 transformer의 장점(Optimizing/long-term dependency)를 완전히 이용하고 있다고 볼 수 없습니다.

이 모델의 평가를 진핼할 때, 위 그림의 오른쪽과 같이 학습과 동일한 컨텍스트 길이로 한 칸씩 옮겨가며 다음 토큰을 예측하는 방식으로 진행합니다.(sliding window) 이 방식은 각 단계에서 학습 때 주어진 최대 길이의 정보를 이용할 수 있지만, 많은 계산량을 요구하고 느린 속도를 갖습니다.

## 3.2 Segment-Level Reccurence with State Reuse

![transformer-xl](/images/Transformer-XL/transformer_xl.png){: width="100%"}{: .center}

Vanilla Transformer의 문제를 해결하기 위해 Transformer 구조의 recurrence 방법을 적용합니다. 학습이 진행되는 동안, 각 시그먼트의 연산 결과들을 다음 시그먼트가 이용할 수 있도록 저장(fixed/cached)합니다. 현재 시그먼트에서는 모델링을 직전 시그먼트의 정보를 이용할 수 있습니다. 즉 위 그림의 왼쪽 부분과 같이 하나의 시그먼트를 모델링 하기 위해 두 개의 연속된 시그먼트의 정보를 이용합니다.

$$\tilde{h}^{n-1}_{\tau+1} = [SG(h^{n - 1}_{\tau + 1}) ; h^{n-1}_{\tau + 1}]$$

$$\tau - 1$$과 $$\tau$$ 시점의 $$n-1$$ 번째 layer의 hidden state를 concat하여 Transformer $$n$$ 번째 layer에서 이용될 입력을 구성합니다. SG는 Stop Gradient로, Backpropagation을 할 때, 이전 시그먼트의 hidden state를 만들기 위해 이용되었던 prameter는 학습되지 않습니다.(Grandient가 전파되지 않습니다.)

$$q^n_{\tau + 1}, k^n_{\tau + 1}, v^n_{\tau + 1} = h^{n - 1}_{\tau + 1}W^T_q, \tilde{h}^{n-1}_{\tau+1}W^T_k, \tilde{h}^{n-1}_{\tau+1}W^T_v$$

첫 번째 식에서 만들어진 hidden state를 이용하여 Self attention에 이용될 Key, Query, Value 벡터를 만듭니다. 이 때, 현재 시그먼트에 대한 hidden state를 계산하면 되기 때문에 Query는 $$h^{n - 1}_{\tau + 1}$$를 이용하여 계산됩니다.

$$h^n_{\tau + 1} = Transformer\_Layer(q^n_{\tau + 1}, k^n_{\tau + 1}, v^n_{\tau + 1})$$

만들어진 Query, Key, Value를 이용하여 현재 Segment에 대한 현재 Layer의 hidden state($$h^n_{\tau + 1}$$)를 만듭니다.

이와 같이 연속된 두 시그먼트 사이에 recurrence 매커니즘이 **시그먼트 단위**로 적용되었습니다. 결과적으로 $$t$$ 시점의 시그먼트는 $$ < t$$ 시점의 정보들 또한 포함할 수 있습니다. 하지만 $$h^n_{\tau+1}$$은 $$h^{n-1}_{\tau}$$를 이용하는 의존관계 때문에 시그먼트 하나 당 하나의 layer씩 밀리게 됩니다. (동일한 layer에서 recurrence연결이 있는 RNN과 차이가 있습니다.) 따라서 최대 의존관계는 시그먼트 length $$\times$$ layer 수로 제한됩니다. 위 그림의 오른쪽의 경우에는 $$4(Segment \space length) \times 3 (layer)$$로 최대 12길이의 의존성을 가질 수 있습니다.

평가시 속도 측면에서 보면, 이전 시그먼트의 state를 저장함으로써, sliding window 방식(vanilla transformer)을 이용하지 않아도 되기 때문에, 기존의 방법보다 약 +1800배 빠르게 연산을 진행할 수 있습니다. 또한 평가 환경에서 장치의 메모리 등이 허용하는 한에서 이전 시그먼트의 길이를 더 길게 가져감으로써, 더 긴 의존성을 만들 수 있습니다. 이 때 이전 시그먼트는 memory augmented neural network의 메모리와 유사한 역할을 수행할 수 있습니다.

## 3.3 Relative Positional Encodings

앞에서 제시한 구조를 실현하기 위해서는 한가지 문제를 더 해결해야하는데, Positional Encoding 입니다. 기존의 Transformer는 특정 위치의 토큰의 embedding을 계산할 때, 해당 토큰의 Word Embedding과 토큰 위치의 Positional Encoding값을 더합니다. 따라서 각 위치마다 고유한 embedding 값을 갖습니다. 이를 recurrence 매커니즘에 적용해보면, 다음과 같습니다.

$$h_{\tau + 1} = f(h_{\tau}, E_{s_{\tau + 1} + U_{1:L}})$$

$$h_{\tau} = f(h_{\tau - 1}, E_{s_{\tau} + U_{1:L}})$$

$$E$$ 는 word embedding $$U$$ 는 Positional encoding입니다. 이 때 이전 state를 만들 때와 현재 state를 만들 때 사용한 positional encoding($$U$$)이 동일하여, 모델이 $$x_{\tau, j}$$ 와 $$x_{\tau + 1, j}$$ 를 구분할 수 없습니다.

positional encoding은 모델에게 토큰의 위치에 대한 단서/bias(어디에 attend 해야 하는지)를 제공합니다. 동일한 목적을 위해서는 위치 정보를 일반적인 transformer의 방식과 같이 초기 임베딩에 포함시키는 것 대신 각 layer의 attention score에 직접 포함시킬 수 있습니다. 또한 attention에서는 각 토큰의 query, key 벡터 사이의 유사도를 계산하는데, 이 때 각 토큰의 절대적 위치를 아는 것 보다는 두 토큰 사이의 상대적인 거리를 아는 것이 더 중요합니다. 즉 시간적 정보를 "절대적"(기존의 방식)이 아닌 "상대적"으로 정의하는 것이 조금 더 직관적이고, 일반화 가능합니다.

이러한 근거에 따라 저자들은 기존 Transformer의 방식에서 부터 새로운 Relative Positional Encoding 방식을 제안합니다. 각 토큰의 위치를 나타내는 절대적 인코딩 대신 두 토큰 사이의 거리를 나타내는 0~$$L_{max}$$사이의 상대적 인코딩을 만듭니다. 먼저 기존의 Transformer의 attention 계산식부터 살펴보면 다음과 같습니다.

$$A_{i, j}^{abs} = (E_{x_i}^T + U_i^T)W_q^T((E_{x_j}^T + U_j^T)W_k^T)^T$$

절대적인 postional encoding을 이용했을 경우, $$i$$ 번째 query 토큰과 $$j$$번째 key 토큰 사이의 ateention 값은 위 식과 깉이 계산할 수 있습니다. 이 식을 풀어쓰면 다음과 같습니다.

$$A_{i, j}^{abs} = \underbrace{E_{x_i}^TW_q^TW_kE_{x_j}}_{(a)} + \underbrace{E_{x_i}^TW_q^TW_kU_j}_{(b)} + \underbrace{U_i^TW_q^TW_kE_{x_j}}_{(c)} + \underbrace{U_i^TW_q^TW_kU_j}_{(d)}$$

총 4가지 텀으로 구성되어 있고, 위 식에 Relative Positional Encoding을 적용하여 다음과 같은 형태로 나타냅니다.

$$A_{i, j}^{rel} = \underbrace{E_{x_i}^TW_q^TW_{k, E}E_{x_j}}_{(a)} + \underbrace{E_{x_i}^TW_q^TW_{k, R}R_{i-j}}_{(b)} + \underbrace{u^TW_{k, E}E_{x_j}}_{(c)} + \underbrace{v^TW_{k, R}R_{i-j}}_{(d)}$$

기존의 식과 차이점을 살펴보면 다음과 같습니다.
- 절대적인 positional encoding, $$U_i, U_j$$를 상대적인 $$R_{i-j}$$로 변경했습니다. $$R \in \mathbb{R}^{L_{max} \times d}$$로, $$i$$ 번째 행은 상대적인 거리가 $$i$$인 위치의 encoding 값입니다.
- $$U_i^TW_q^T$$를 $$u \in \mathbb{R}^d$$로 변경했습니다. query 벡터는 모든 위치에 대해 동일한 값을 가지는데, attention 값을 계산하기 전에 서로 다른 단어들이 위치에 관계없이 일정하게 가지는 bias로 볼 수 있습니다. (위 식에서 (a), (c)텀을 묶어서 보면 $$(E_{x_i}^TW_q^T + u^T)W_{k, E}E_{x_j}$$과 같이 나타내어 집니다.)
- 비슷한 이유로 $$U_i^TW_q^T$$를 $$v \in \mathbb{R}^d$$로 대체합니다. ((b), (d)텀을 묶어서 보면 $$(E_{x_i}^TW_q^T + v^T)W_{k, R}R_{i-j}$$로 나타내어 집니다.)
- 마지막으로 key value에 대한 Wiehgt를 $$W_{k, E}, W_{k,R}$$로 나누는데 각각 컨텐츠 기반, 위치 기반의 key 벡터를 만듭니다. $$A_{i, j}^{rel} = (E_{x_i}^TW_q^T + u^T)W_{k, E}E_{x_j} + (E_{x_i}^TW_q^T + v^T)W_{k, R}R_{i-j}$$과 같이 나타내면 명확히 각 $$W_k$$의 역할을 구분할 수 있습니다.

각 텀들의 직관적 의미는 다음과 같습니다.
- (a): 컨텐츠 기반의 전달(addressing)
- (b): 컨텐츠에 의존하는 positional bias
- (c): 글로벌 컨텐츠에 대한 bias
- (d): 글로벌 위치에 대한 bias

recurrence 매커니즘이 적용된 Transformer-XL 각 layer의 동작은 다음과 같습니다.(아래 식의 attention은 sinlge attention head만 표현합니다.)

$$\tilde{h}^{n-1}_{\tau} = [SG(m^{n - 1}_{\tau}) ; h^{n-1}_{\tau}]$$

$$q^n_{\tau}, k^n_{\tau}, v^n_{\tau} = h^{n - 1}_{\tau}W^{n\top}_q, \tilde{h}^{n-1}_{\tau}W^{n\top}_{k, E}, \tilde{h}^{n-1}_{\tau}W^{n\top}_v$$

$$A^n_{\tau, i, j} = q^{n\top}_{r, i} k^n_{\tau, j} + q^{n\top}_{r, i} W^n_{k,R}R_{i - j} + u^{\top}k_{\tau, j} + v^{\top}W^n_{k,R}R_{i-j}$$

$$a^n_{\tau} = Masked\_Softmax(A^n_{\tau})v^n_{\tau}$$

$$o^n_{\tau}=LayerNorm(Linear(a^n_{\tau}) + h^{n-1}_{\tau})$$

$$h^n_{\tau} = Positionwise\_Feed\_Forward(o^n_{\tau})$$

Attention 연산을 제외한 전체적인 알고리즘은 Transformer와 동일합니다. 가장 초기 입력은 $$h^0_{\tau}:= E_{s_{\tau}}$$으로 단어 임베딩 값으로만 구성됩니다. 실제 [저자들의 구현체](https://github.com/kimiyoung/transformer-xl/tree/master/pytorch)를 살펴보면, attention의 종류와 positional encoding의 종류(sinusoidal, learnable 등)에 따라 여러 모듈들이 존재합니다. 모든 토큰 쌍 $$(i,j)$$에 대해 positional encoding을 계산하는 $$W^n_{k, R}R_{i - j}$$는 quadratic의 cost를 요구하기 때문에 이를 빠르게 계산하는 방법 또한 Appendix + 구현에서 제공합니다.

## 4. Experiments

## 4.1 Main Result

저자들은 다영한 word 단위, character 단위의 language modeling 데이터셋들(WikiText-103, enwik8, text8, One Billion Word)로 학습을 진행했고, SOTA모델들과 비교했습니다.

![wikitext](/images/Transformer-XL/wikitext.png){: width="50%"}{: .center}

WikiText-103은 이용가능한 가장 큰 word단위의 language modeling 벤치마크이고, long-term dependency를 갖고 있습니다. 총 28K의 기사들로 부터 만들어진 103M의 학습 토큰으로 구성됩니다. 또한 평균적으로 기사 하나당 3.6K의 토큰이 존재해서, long-term dependancy 모델링 틍력을 테스트할 수 있습니다. 학습 동안은 384, 평가 동안은 1600의 attention 길이를 이용했습니다. 위의 표와 같이 이전 SoTA ppl을 20.5에서 18.3로 능가함을 볼 수 있습니다.

![enwik](/images/Transformer-XL/enwik.png){: width="50%"}{: .center}

enwik8은 100Mb의 정제되지 않은 위키피디아 텍스트입니다. 학습 동안 784, 평가 동안 3800의 attention length를 이용했습니다. vanilla transformer뿐만 아니라, RNN-based 모델은 큰 격차로 뛰어넘음을 볼 수 있습니다. 또한 Vanilla transformer-64layer의 성능과 transformer-xl-12layer(약 17%의 크기)이 동일하고, xl또한 layer수를 늘릴 수록 더 좋은 성능을 보입니다. 보조의 loss들을 이용하지 않고 기존의 모델들을 넘었기 때문에 더 좋은 구조 덕분이라고 볼 수 있습니다.

![text8](/images/Transformer-XL/text8.png){: width="50%"}{: .center}

text8은 enwik8과 달리 100M의 정제된 위키피디아 데이터 입니다. a-z의 character만 포함하며, 모두 소문자로 구성됩니다. enwik8에서 가장 좋았던 하이퍼파라메터 설정으로 학습을 진행했고, SoTA성능을 달성했습니댜.

![one_billion_word](/images/Transformer-XL/one_billion_word.png){: width="50%"}{: .center}

One Billion Word 데이터셋은 문장들을 섞었기 때문에, long-term dependency를 보존하지 않습니다. 즉, 주로 short-term dependency를 모델링하는 것을 테스트하는 데이터셋입니다. Transformer-XL은 long-term dependency학습에 중점을 두었음에도 불구하고, SoTA를 달성했고 short-term dependency도 잘 일반화한다고 볼 수 있습니다.

## 4.2 Ablation Study

초기에 기존 방법들이 갖고 있던 두 가지 문제 1)fixed-length로 long-term dependency에 제약 2) context fragmentation 문제를 제시했습니다. 이 문제들에 대해 Transformer-XL에서 이용한 두 가지 메인 테크닉 1)recurrence 구조 2)relative positional encoding 의 효과를 증명합니다.

첫번 째 실험은 WikiText-103을 이용했는데, 이 데이터셋은 위에서 언급 했듯이 long-term dependency 모델링을 테스트하기에 적합합니다. 표의 각 항목들은 다음과 같습니다.

![ablation_1](/images/Transformer-XL/ablation_1.png){: width="100%"}{: .center}

- **`Recurrence`**
  - True: 주어진 Attention length만큼 Recurrence하게 연산을 진행하는 방식
  - False: Recurrence연결 없이 기존의 sliding window 방식
- **`Encoding`**
  - 상대적인 방식: 본 논문에서 제시한 방법, Shaw et al. (2018)
  - 절대적인 방식: Vaswani et al.,(2017), Al-Rfou et al.(2018)
- **`Loss`**
  - half: 마지막 반의 위치에 있는 토큰들에 대해서만 loss를 계산하는 방식
  - full: 모든 위치에 있는 토큰들에 대해 loss를 계산하는 방식으로 구성됩니다.
- **`PPL init`**: 학습시 같은 길이를 사용했을 때의 PPL
- **`PPL best`**: 최적의 길이를 이용했을 때의 PPL
- **`Atth Len`**: PPL best를 얻기 위해 이용된 attention length 입니다.

위 표의 결과를 보면 제시한 두가지 테크닉이 모두 퍼포먼스에 영향을 준다는 것을 알 수 있습니다.
- `Encoding`을 제외한 모든 조건이 동일한 5번째 실험 vs (8,9 번째 실험)을 보면, 상대적인 positional encoding 방식이 더 우세한 성능을 보입니다.
- `Recurrence`를 제외한 모든 조건이 동일한 2번째 실험 vs 6번째 실험을 비교해보면 Recurrence 구조가 더 좋은 성능을 보입니다.
- 마지막 3개의 실험을 비교해보면 `Attn Len`이 클수록 더 좋은 성능을 보입니다.

두번 째 실험은 더 긴 컨텍스트를 볼 수 있는 장점을 통해 context fragmentation 문제를 해결했는지 확인하기 위한 실험을 진행했습니다. 따라서 비교적 long-term dependency를 덜 요구하는 One Billion Word 데이터셋을 이용했습니다.

![ablation_2](/images/Transformer-XL/ablation_2.png){: width="50%"}{: .center}

위 표와 같이 long-term dependency가 중요하지 않은 테스트에서도 recurrence를 이용한 경우에 더 좋은 성능을 얻었습니다. 또한 짧은 시퀀스에 대해 Shaw et al.,(2018)의 인코딩 방식 보다 뛰어난 성능을 보였습니다. 결과적으로 recurrence 구조는 context fragmentation 문제를 해결한다고 볼 수 있습니다.

# 5. Appendix - Efficient Computation of the Attention with Relative Positional Embedding

모든 Query, Key 토큰 쌍 (i,j)에 대해 $$W_{k, R}R_{i - j}$$를 계산하면, 나이브한 방식으로는 quadratic의 비용을 갖습니다. 따라서 linear의 cost를 갖는 방법을 소개합니다.

$$i-j$$의 범위는 0부터 $$M + L - 1$$까지 입니다. ($$M$$: Memory 길이, $$L$$: Segment 길이) 따라서 $$R = \{R_0, ... R_{M+L-1}\}$$ 입니다.

![efficient_1](/images/Transformer-XL/efficient_1.png){: width="70%"}{: .center}

위 방식으로 $$W_{k, R}R_{i - j}$$의 가능한 모든 쌍에 대한 계산을 진행합니다. 이 때, $$Q$$는 크기의 역순으로 정의되어 $$Q_k = W_{k, R}R_{M+L-1-k}$$와 같은데, 이를 통해 이후 연산을 조금 더 쉽게 진행할 수 있습니다.

![efficient_2](/images/Transformer-XL/efficient_2.png){: width="100%"}{: .center}

relative positional encoding의 (b) 텀,  $$E_{x_i}^TW_q^TW_{k, R}R_{i-j}$$ 을 현재 query, key 쌍에 대해 모두 계산합니다. 위 행렬에서 행은 query, 열은 key로 볼 수 있습니다. query는 Segment로 $$L$$만큼의 길이를,  key는 Memory + Segment로 총 $$M + L$$의 길이를 갖습니다. 행렬의 각 위치에서 query와 key의 상대적인 거리에 따라 $$R_{i-j}$$값이 결정됩니다.

![efficient_3](/images/Transformer-XL/efficient_3.png){: width="80%"}{: .center}

첫번 째 그림에서 만들었던 $$Q$$와 query 벡터 $$E^TW_q^T$$를 곱하면 $$\tilde{B}$$를 얻을 수 있습니다. $$\tilde{B}$$의 각 행을 왼쪽으로 shift하면, $$\tilde{B}$$와 동일한 행렬을 만들 수 있습니다. 결과적으로 $$qQ^{\top}$$의 행렬곱과 `left-shift` 연산을 진행하면 (b) 텀을 계산할 수 있습니다.

# 7. Reference

- Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov. Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context. In ACL, 2019.

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems(NeurIPS), 2017.


# 수정해야할 사항

- Transformer Decoder를 이용했다는 점 추가
- 각 layer별로 positional encoding 정보를 계속해서 추가해준다는 점 추가