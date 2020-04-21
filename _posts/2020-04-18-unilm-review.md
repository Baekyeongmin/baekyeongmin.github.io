---
title: "UniLM Review"
layout: post
categories:
  - paper-review
tags:
  - paper
  - nlp
last_modified_at: 2020-04-18T20:53:50-05:00
author: yeongmin
comments: true
---

Pre-trained LM은 많은 양의 데이터로 부터 컨텍스트를 반영한 representation을 학습합니다. 이를 통해 NLU-`BERT`, `RoBERTa`, NLG-`GPT` 와 같이 각각의 downstream 테스크에서 좋은 성능을 보여주고 있습니다. 이번 포스트에서는 NLU와 NLG를 함께 pre-train하여 하나의 pre-trained 모델이 각 테스크에 "모두" fine-tuning될 수 있는 방법인 "**Uni**fied pre-trained **L**anguage **M**odel(UniLM)"를 제시한 논문 ["Unified Language Model Pre-training for Natural Language Understanding and Generation (NeurIPS 2019)"](https://arxiv.org/abs/1905.03197)를 리뷰하려고 합니다.

# 1. Main Idea

- 마스킹된 토큰을 예측하는 pre-training 테스크와 어텐션 마스킹 방식을 조합하여 3가지 Language Modeling Objective(`Uni-directional`, `Bi-directional`, `Sequence-to-Sequence`)를 학습합니다.
- 각 LM 테스크 사이의 파라메터와 모델 구조를 단일 Transformer로 통일함으로써, 여러 LM을 만들어야 했던 필요성을 완화합니다.
- 각 objective 사이의 파라메터 공유를 통해 컨텍스트를 여러 방향으로 이용할 수 있는 능력을 학습하고, 오버 피팅을 방지하여 일반화 능력을 향상시킵니다.
- NLU테스크 뿐만 아니라 Sequence-to-Sequence LM으로써, 요약이나 질문 생성 등의 생성에 이용할 수 있습니다.

# 2. Unified Language Model Pre-training

![unilm](/images/unilm/unilm.png){: width="100%"}{: .center}

입력으로 토큰들의 시퀀스가 주어졌을 때, UniLM은 Transformer 구조를 이용해 컨텍스트를 고려한 각 토큰의 representation을 계산합니다. 이 때 제시한 세가지 objective를 수행하기 위해, 위 그림과 같이 어텐션 마스크를 이용해서 각 토큰이 어떤 컨텍스트에 접근할지 통제합니다. 즉, 얼마나 많은 토큰들(컨텍스트)에 attend 할지를 결정합니다.

## 2.1. Input Representation

- 입력은 1개 혹은 2개의 시그먼트(연속된 토큰의 시퀀스)로 구성됩니다. unidirectional LM의 경우 1개의 시그먼트를 입력으로 하고, 나머지 objective는 2개의 시그먼트를 입력으로 합니다.
- 모든 시퀀스는 `[SOS]`(start-of-sequence) 토큰으로 시작하고, 각 시그먼트는 `[EOS]`(end-of-sequence) 토큰으로 끝납니다. `[EOS]`토큰은 각 시그먼트의 경계 역할 뿐만 아니라 NLG에서 디코딩 과정의 끝을 학습할수 있게 합니다.
- 모든 텍스트는 WordPiece에 의해 서브워드로 토크나이즈 됩니다.
- 각 입력 토큰의 임베딩은 BERT와 유사하게 token, position, segment 임베딩의 합으로 계산됩니다. 서로 다른 objective를 임베딩 단에서 구분하기 위해, objective마다 다른 segment임베딩을 사용합니다.

## 2.2 Backbone Network: Multi-Layer Transformer

각 토큰의 입력 벡터는 L개의 layer를 가진 Transformer 블럭을 거치면서, 컨텍스트를 반영한 representation을 만듭니다. 각 블럭은 일반적인 Transformer와 동일하게 multi-head self-attention 연산을 수행합니다.

$$Q=H^{l-1}W_l^Q, K=H^{l-1}W_l^K, V=H^{l-1}W_l^V$$

$$M_{ij} = \begin{cases} 0, \text{allow to attend} \\ -\infty, \text{prevent from attending} \end{cases}$$

$$A_l = softmax(\frac{QK^{\top}}{\sqrt{d_k}} + M)V_l$$

Self-Attention 연산은 위의 식과 같이 이루어지는데, 주목해야할 점은 2번째 식의 행렬 $$M$$입니다. 3번째 식에서 attention value(Q-K사이의 관계)에 $$M$$을 더해서 softmax연산을 진행하는데, 이 때 $$M=-\infty$$ 인 경우 해당 Key에 attend할 비율은 0이 되어 버립니다. 일반적으로, BERT에서는 패딩 값들, GPT와 같은 생성모델에서는 현제 시점에서 미래 위치의 토큰값들에 attend하는 것을 막기 위해서 $$M$$을 적절히 이용합니다. (본 논문에서 사용법은 아래의 pre-training objective에서 자세히 다룹니다.)

## 2.3. Pre-training Objective

UniLM은 cloze 테스크(빈칸 맞추기-마스킹된 토큰 맞추기)를 수행하는데, 4가지의 서로 다른 LM objective에 따라 **attend할 수 있는 컨텍스트**가 달라집니다. 즉 각 Objective가 동일한 문제(cloze 테스크)를 풀되, 마스킹된 토큰을 맞추기 위해 주어지는 정보량에 차이가 있습니다. cloze 테스크는 BERT의 MLM과 유사한 방식으로 진행합니다. 입력에서 랜덤으로 몇몇 토큰을 `[MASK]`로 치환하고 Transformer 모델의 최종 출력값을 이용해 원래 토큰을 예측합니다.

- **Unidirectional LM**: `left-to-right`, `right-to-left` 두 방향의 컨텍스트를 각각 이용하여 마스킹된 토큰을 예측하는 단방향 Language Modeling을 학습합니다. `left-to-right`의 경우 주어진 토큰 $$[x_1, x_2, [MASK], x_4]$$에서 $$[MASK]$$를 예측하기 위해 왼쪽의 컨텍스트 $$x_1, x_2$$와 자기자신($$[MASK]$$)만을 이용합니다. 이를 위해서 아래 그림과 같이 각 토큰의 attention을 계산할 때, 해당 시점 이전(자기 자신 포함)의 토큰들만 이용하도록 마스크를 구성합니다.

![ltor](/images/unilm/ltor.png){: width="50%"}{: .center}

- **Bidirectional LM**: BERT의 방식과 동일하게, 양방향의 컨텍스트를 모두 이용하여 마스킹된 토큰을 예측하는 Language Modeling을 학습합니다. 아래 그림과 같이 모든 마스크의 값이 0으로, 각 토큰은 서로를 attend할 수 있습니다. 이를 통해 양 방향으로부터 더 많은 컨텍스트 정보를 인코딩할 수 있게 되고, 단 뱡향보다 풍부한 표현력을 갖게 됩니다.

![bid](/images/unilm/bid.png){: width="50%"}{: .center}

- **Sequence-to-Sequence LM**: Sequence-to-Sequence 방식은 두 시그먼트가 각각 소스/타겟의 의미를 갖습니다. 전형적인 Seq2Seq과 유사하게 소스의 토큰들은 각 토큰들이 서로 attend할 수 있고(인코더의 역할), 타겟의 토큰들은 해당시점 이전의 토큰들(소스 토큰 + 현재 시점 이전의 타겟 토큰들, 자기자신 포함)에 attend 할 수 있습니다.(디코더의 역할)

    주어진 입력 시퀀스 $$[SOS], t_1, t_2, [EOS], t_3, t_4, t_5, [EOS]$$ 에 대해 첫번째 시그먼트 ($$[SOS], t_1, t_2, [EOS]$$)는 서로 attend 할 수 있고 두번째 시그먼트의 $$t_4$$같은 경우 자기자신 포함 왼쪽의 6토큰에 attend할 수 있습니다.

    아래 그림과 같이 소스는 타겟 시그먼트를 볼 수 없도록 마스킹되고(우측 상단), 타겟은 현재 시점 이후의 토큰을 볼 수 없도록 마스킹 됩니다.

![s2s](/images/unilm/s2s.png){: width="50%"}{: .center}

이렇게 마스킹이 된 입력의 원래 토큰을 예측하는 테스크를 "서로 다른 마스킹 기법"으로 4가지