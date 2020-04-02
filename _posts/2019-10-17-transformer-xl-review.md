---
title: "Trnasformer-XL Review"
layout: post
categories:
  - paper-review
tags:
  - paper
  - nlp
last_modified_at: 2019-10-17T20:53:50-05:00
author: yeongmin
comments: true
---


이번 글에서는 ACL 2019에서 발표된 ["Trnasformer-XL: Attentive Language Models Beyond a Fixed-Length Context"](https://arxiv.org/abs/1901.02860)를 리뷰하려고 합니다. 본 논문은 기존의 Transformer 구조를 이용한 고정된 길이(Fixed-Length) Language Model의 한계점을 지적하고 더 긴 Long-term dependancy를 이용할 수 있는 새로운 방법을 제시합니다. 또한 다양한 NLU 테스크들에서 SOTA성능을 보이고 있는 [XLNet](https://arxiv.org/abs/1906.08237)과 동일한 저자들이 작성하였고, Transformer-XL의 많은 부분을 XLNet에서 이용하고 있습니다.

# Main Idea
1. 기존의 RNN(Reccurent Neural Network)-LM(Language Model)은 RNN자체의 문제인 Vanishing/exploding gradient, long-term denpendency등의 문제를 갖고 있었었습니다. Transformer 구조는 이와 같은 문제 및 RNN의 한계점들을 잘 해결하여 Sequential modeling에 새로운 방식을 제시했습니다. 하지만 Trnasformer 기반의 LM도 한계점을 갖는데, 본 논문에서는 이를 해결할 수 있는 새로운 방법을 제시합니다.

2. 기존의 Transformer기반의 LM(vanilla transformer model)은 코퍼스를 여러 개의 segment들로 나누고, 아래 그림과 같이 각 segment별로 *"해당 segment내에서"* Langauge Modeling의 Auto-regressive한 Objective를 학습했습니다. 즉 각 segment의 고정된 최대 길이 내에서만 학습이 이루어지므로, 해당 범위를 벗어나는 long-term dependancy는 학습할 수 없습니다. 또한 각 segment는 문장 등의 의미있는 단위로 나눠진 것이 아닌 단순하게 연속적인 symbol들(token, word 등)의 조각들로 구성되기 때문에 해당 segment의 처음 몇개의 symbol들을 예측하기에는 필요한 정보의 양이 부족한 문제가 발생합니다. 저자들은 이러한 문제들을 해결하기 위해 Trnasformer-XL(extra long)이라는 방법을 제시합니다.

![vanilla-Transformer](/images/Transformer-XL/vanilla.jpg){: width="100%"}{: .center}

# Abstract

- Transformer 구조는 장기 의존성을 배울 수 있는 가능성을 갖고 있지만, language modeling의 환경에서 고정된 길이의 context에 제한된다.
- Transformer-XL이라는 고정된 길이 이후의 의존성을 학습할 수 있는 방법을 제시함.
- 시그먼트 단위의 recurrence 메커니즘과 positional encoding으로 구성된다.
- 장기 의존성을 인식할 수 있을뿐만 아니라, 컨텍스트 분열 문제를 막아준다. 결과적으로 transformer XL은 RNN보다 80%긴 의존성과 vanilla Transformer보다 450% 더 긴 의존성을 학습한다.
- evaluation 단계에서 vanilla transformer보다 1800배 이상 빠른 성능을 보여준다.
- bpc/perplexity의 결과에서 SOTA보다 향상된 결과를 보였다.

# Introduction

- Language Modeling은 unsupervised pretraining과 같은 성공적인 적용법(ELMo, GPT)과 함께 장기 의존성을 모델링하는 것을 요구하는 중요한 문제이다. 하지만 시퀀셜한 데이터에 장기 의존성 모델링 능력을 뉴럴넷으로 학습하기는 상당히 챌린징하다.
- RNN은 표준적인 솔루션이였다. 많은 적용에도 불구하고 RNN은 옵티마이즈하기 어려웠고, (gradient/explosion vanishing) LSTM과 gradient clipping등의 테크닉이 이용되었으나 이걸로 문제를 다 해결하기에 충분하진 않았다.
- 경험에 따르면, 이전 연구들은 LSTM 언어 모델은 평균적으로 200개의 단어를 컨텍스트로 이용하였다.
- 반면에, 긴 길이를 갖고 있는 단어 페어 사이의 직접적인 연결을 갖고있는 어텐션 메커니즘은 옵티마이즈가 쉽고, 장기 의존성을 학습할 수 있다.
- 최근에 깊은 캐릭터 레벨의 언어 모델링에서 transformer를 학습하기 위한 보조의 로스를 디자인하는 연구들이 있었는데, LSTM을 큰 격차로 뛰어 넘었다. 그럼에도 불구하고 몇 백개의 고정된 길이의 분리된 시그먼트로 학습되었다.(시그멘트 사이의 정보 전달 없이)
- 고정된 컨텍스트 길이 때문에, 모델은 이 길이보다 긴 의존성을 학습하지 못했다. 추가적으로, 각 시그먼트는 문장 혹은 어떤 의미적인 바운더리 에 관계없이 연속된 토큰들이 선택되었다. 따라서 모델이 처음 몇 토큰을 예측하기 위한 contexual 정보가 부족했다. -> context fragmentation
- 이러한 문제를 해결하기 위해 Transformer-XL이라는 구조를 소개한다. 새로운 시그먼트 마다 다시 히든 스테이트를 계산하는것 대신에, 이전 시그먼트에서 얻어진 히든 스테이트를 재사용한다. 재사용되는 히든 스테이트는 현재 시그먼트를 위한 메모리를 제공한다. 시그먼트들 사이에 recurrent한 연결을 제공한다. -> 이 연결을 통해 정보가 propagate되고, 매우 긴 의존성을 모델링할 수 있게된다.
- 이전 시그먼트로부터 정보를 넘겨주는것은 또한 Context fragmentation 문제도 해결한다.
- 절대적인 positional encoding(transformer의 방식)대신에 relative positional encoding이 필요함을 보인다.
- transformer의 recurrence + positional encoding이 주요 컨트리뷰션

# Model

t시점 이하의 토큰들이 주어졌을 때, 다음 토큰을 예측하는 conditional probability를 추정하는 전형적인 language modeling task를 푼다.