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
