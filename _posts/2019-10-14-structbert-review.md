---
title: "StructBert Review"
layout: post
categories:
  - paper-review
tags:
  - paper
  - nlp
last_modified_at: 2019-10-14T20:53:50-05:00
author: yeongmin
comments: true
---

이번 글에서는 Alibaba Group에서 연구한 [StructBERT: Incorporating Language Structures into Pre-training for Deep Langauge Understanding](https://arxiv.org/pdf/1908.04577.pdf)를 리뷰하려고 힙니다. 본 논문은 현재 ALBERT에 이어 GLUE 벤치마크 리더보드에서 89.0(2위)의 점수를 얻고 있습니다. 언어의 유창함은 단어와 문장의 순서에 의해 결정되는데, 본 논문에서는 sequencial modeling은 "순서"에 대해 주목했습니다. BERT pre-training 단계에서 문장 내부와 문장들 사이 순서의 구조적 정보를 학습할 수 있는 새로운 전략을 제시합니다. 

# Main Idea
1. BERT에서 Bidirectional Transformer Encoder를 이용하여 두 개의 Objective(Masked-LM, NSP)를 pre-training하여 각 토큰의 Contextual representation을 얻을 수 있었습니다. 본 논문에서는 pre-training 단계에서 특정 토큰들 및 문장들의 "순서"를 예측하는 문제를 추가함으로써, 모델이 잘 정제된 단어 구조와 문장간의 구조를 학습하도록 했습니다.

2. 이상적으로 잘 학습된 Language model은 랜덤한 순서의 단어들이 주어졌을 때, 이를 복구할 수 있어야 하는데, BERT의 학습전략으로는 단어들의 sequential 순서와 단어들의 고차원 의존성을 정확히 모델링하지 못한다고 언급합니다. 이러한 문제를 해결하기 위한 새로운 단어 단위의 pre-training objective를 제시합니다.

3. NSP objective는 쉽게 97-8% 정도의 정확도에 도달할 수 있는데, 이는 이 문제 자체가 너무 쉽기 때문입니다. 이러한 점에서 NSP를 확장하여 조금 더 어려운 문장들(segment) 사이의 관계를 모델링할 수 있는 pre-training objective를 제시합니다.

# StructBERT
## Model Architecture

기본적으로 StructBERT는 입력 문장의 contextual representation을 인코딩하기 위해, Multi-Layer Transformer Encoder, 즉 BERT의 모델 구조를 수정없이 그대로 이용합니다. 입력의 형태또한 BERT의 방식과 동일하게 `[CLS]` + Segment1 + `[SEP]` + Segment2 + `[SEP]`를 이용합니다. 두 segment를 구분하기 위한 segment embedding 및 토큰의 position을 구분하기 위한 positional embedding역시 동일하게 이용합니다.

## Word Structural Objective

![Pre-training-objective](/images/StructBERT/pre_training_task.jpg){: width="100%"}{: .center}

특정 범위 내에서 임의로 섞여진 토큰들을 원래의 순서로 복구하는 문제를 풀고자 합니다. Masked-LM이 임의로 토큰을 Masking하고 classifier가 `[MASK]` 토큰의 위치에서 원래 토큰의 likelihood가 최대가 되도록 학습한다면(위 그림의 노란색), 이 방법은 임의로 섞여진 토큰들이 주어지고, classifier는 해당위치에서 올바른 토큰의 likelihood가 최대가 되도록하는 학습합니다.(위 그림의 파란색) 즉 이 obejctive는 $$arg \max_{\theta} \sum \log P(pos_1 = t_1, pos_2 = t_2, ...pos_K = t_K \mid t_1, t_2, .... , t_K, \theta)$$ 와 같이 나타낼 수 있습니다. 여기서 *K*는 섞여지는 subsequence의 길이인데, K가 클수록 모델은 순서가 더 많이 어그러진 문장을 복구하게 됩니다. 본 논문에서는 K=3으로 하여 최대 trigram 내부에서만 순서를 섞었습니다. 이 문제는 다음과 같은 과정으로 진행됩니다.

1. 입력 토큰들의 15%를 masking하고 이를 Transformer Encoder로 인코딩 한 후, 이를 이용하여 원래의 토큰을 예측합니다.
2. masking이 되어 있지 않은 trigram중 일정 부분을 랜덤으로 선택하여 해당 토큰들을 섞습니다.
3. 이렇게 섞여진 입력을 다시 Transformer Encoder로 인코딩 한 후, softmax classifier로 원래 토큰 순서를 예측합니다. 위 그림과 같이 $$(t_3, t_4, t_2)$$ 가 주어지고 각 위치에서 $$t_3 \rightarrow \t_2, t_4 \rightarrow \t_3, t_2 \rightarrow \t_r$$를 예측하도록 합니다.

Masked-LM Objective와 Word Structural Objective는 동일한 모델로 한번에 함께 학습되며(jointly learned), 동일한 가중치로 학습됩니다. (최종 loss에 반영비율이 같음.)

