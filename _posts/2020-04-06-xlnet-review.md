---
title: "XLNet Review"
layout: post
categories:
  - paper-review
tags:
  - paper
  - nlp
last_modified_at: 2020-04-06T20:53:50-05:00
author: yeongmin
comments: true
---

[지난 포스트](https://baekyeongmin.github.io/paper-review/transformer-xl-review/)에서 "Transformer-XL"에 대한 리뷰를 진행했었는데요. Language Modeling 테스크에서 장기 의존성 능력을 높이기 위해, Transformer의 제한된 컨텍스트 길이를 recurrence한 구조로 늘려주는 방법이였습니다. 이번 포스트에서는 해당 논문의 후속으로 제안된 ["XLNet: Generalized Autoregressive Pretraining for Language Understanding"](https://arxiv.org/pdf/1906.08237)을 리뷰하려고 합니다. 많은 양의 코퍼스로 Language Modeling에 대한 Pre-training을 진행하고 특정 테스크로 Fine-tuning을 진행하는 방법은 BERT 이후로 NLP 문제를 풀기위한 정석과 같은 방법이 되었습니다. XLNet에서는 BERT와 같이 Masked Language Modeling을 objective로 하는 *Autoencoder(AE) 방식*과 GPT와 같이 *Auto-Regressive(AR)* Language Modeling을 objective로 하는 방식의 장점을 유지하면서 단점을 보완하는 새로운 학습 방식을 제안합니다. 또한 Transformer-XL의 recurrence 알고리즘도 함께 적용하여 BERT를 능가하는 성능을 달성합니다. 약 9개월 전에 XLNet 리뷰를 [팀블로그](https://blog.pingpong.us/xlnet-review/)에 작성 했는데, 최근에 논문이 업데이트 되면서 다시 한 번 공부하면서 글을 작성합니다.

# 1. Main Idea

Language Modeling은 특별한 레이블링 작업이 필요 없는 비지도 학습 방식이고, 최근에 언어 자체를 이해하기 위한 pre-training 방법으로 자주 이용됩니다. BERT이전의 방법들은 대부분 Auto-Regressive(AR)방식으로 주어진 컨텍스트에 대해 다음 토큰을 맞추는 **단방향**의 학습을 진행했습니다. BERT에서는 이를 해결하기 위해 특정 토큰을 `[MASK]` 로 치환하고 이를 예측함으로써(Denoising Autoencoder), **양방향**의 정보를 이용할 수 있었습니다. 하지만 1) `[MASK]`는 pre-training 에만 등장하는 토큰으로 fine-tuning 과 불일치 하고, 2) `[MASK]` 토큰 사이의 의존관계가 무시되는 문제가 발생합니다. 본 논문에서는 이를 해결하기 위해, *양방향의 정보를 이용할 수 있는 AR Language Modeling* 학습법을 제안합니다.

# 2. AR, AE Language Modeling

## 2.1. Auto Regressive(AR)

AR Language modeling은 특정 텍스트 시퀀스에 확률을 할당하는 모델입니다. 이를 위해 주어진 텍스트 시퀀스 $$X=(x_1, x_2, x_3,...x_T)$$에 대해 $$t$$ 시점 이하의 토큰 $$x_{<t}$$ 가 주어졌을 때, $$t$$ 시점 토큰의 확률 분포 $$p(x)$$를 추정합니다.

$$p(x) = p(x_t \mid x_{<t}) = \prod\limits_{t=1}^T p(x_t \mid x_{<t})$$ 

즉 ["나는", "블로그", "를", "쓰고", "있다", "."]가 주어졌을 때, $$p(블로그 \mid 나는)p(를 \mid 나는, 블로그) ...$$의 확률 분포를 모델링하도록 학습을 진행합니다.