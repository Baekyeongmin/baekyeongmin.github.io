---
title: "RoBERTa Review"
layout: post
categories:
  - paper-review
tags:
  - paper
  - nlp
last_modified_at: 2019-10-19T20:53:50-05:00
author: yeongmin
comments: true
---

이번 글에서는 ["RoBERTa: A Robustly Optimized BERT Pretraining Approach"](https://arxiv.org/abs/1907.11692)를(GLUE 벤치마크 88.5/4등)리뷰하려고 합니다. Self-Supervised 기반의 학습 방식은 Pre-training에서 많은 시간/리소스가 소요되기 때문에 BERT 및 이후 접근법들을 엄밀하게 비교하기 힘들고, 어떤 Hyper Parameter가 결과에 많은 영향을 미쳤는지 검증하기 힘듭니다. 본 논문에서는 여러 실험을 통해 데이터의 양 및 Key-Hyperparameter의 영향을 분석합니다.

