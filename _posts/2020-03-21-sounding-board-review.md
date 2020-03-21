---
title: "Sounding Board Review"
layout: post
categories:
  - competition
tags:
  - 글또 4기, dualogue
last_modified_at: 2020-03-21T20:53:50-05:00
author: yeongmin
comments: true
---

이전 글에서 Alexa Prize에 대해 간단하게 살펴봤는데, 이 글에서는 Socialbot challenge 2017의 우승작 [Sounding Board – University of Washington’s Alexa Prize Submission](https://m.media-amazon.com/images/G/01/mobile-apps/dex/alexa/alexaprize/assets/pdf/2017/Soundingboard.pdf)을 리뷰해보려고 합니다.

# Sounding Board

- Alexa Prize는 봇이 유저가 선택한 특정 주제에 대해 토론하는 챌린지임 -> 이전의 open-domain chatbot들은 주어진 context에 "적절한" 답을 하는 것을 목적으로 하는 chit-chat에 가까웠음. 하지만 Sounding board에서는 이를 컨텐츠(정보)와 유저의 흥미에 따라 대답을 생성하는 task-oriented 문제로 다루고자 했음.

- UW 팀은 본 대회를 밑바닥 부터 시작했음. 당시 사전에 만들어 두었던 대화 시스템이 없었고, 뉴럴넷을 학습시킬 수 있는 데이터도 없었음. 결과적으로 Sounding Board는 대회가 진행됨에따라 점진적으로 향상되어 갔음. 초기 시스템은 거의 rule-base로 구성 -> 실 유저와의 대화를 경험하고, 이를 통해 대화 데이터를 축적함으로써, 구조는 더 복잡해졌고, 이를 구성하는 요소들은 ML을 사용하도록 변경되어 갔음. (논문에서는 최종 결과물을 다루고자 함.)

- Sounding Board의 디자인 철학은 대화 전략과 시스템 엔지니어링에 반영되어 있음.

    대화의 전략은 두개의 피쳐로 구성.

    - contents driven: 유저가 알지 못하거나 들어보지 못했을 관점의 정보를 제공함으로써 유저를 사로잡고자 했음. -> 정보 검색이 시스템에서 중요한 역할을 함. 다양한 범위의 토픽, 유저의 흥미를 커버하기 위해 여러 정보 소스를 이용했음.
    - user driven: 시스템은 토픽의 선택, 인터렉션 전략 등을 조정하기 위해, 유저의 상태를 트래킹함.
        - 유저 발화의 감정, 스탠스(태도, 입장), 의도 등의 다차원 표현으로 나타냄.
        - 토픽 선택을 위해 성격 퀴즈를 이용함.
        - 사용자의 불만을 감지하여 토픽을 변경함.
    - 대회를 진행하면서 얻은 테스트 유저의 피드백, 유저 행동 분석은 아키텍쳐 결정에 많은 영향을 미침.

    시스템 엔지니어링 전략은 학습을 위한 적절한 대화 데이터의 부족과 여러개의 컨텐츠 소스들을 이용하고자 하는 계획이 주로 고려되었음. 이에 따라 모듈화된 구조를 통한 계층적인 대화 관리를 하는 전략을 선택했음.
    - 이러한 구조는 새로운 능력이 개발되었을 때, 상대적으로 간단하게 추가할 수 있음. -> 더 많은 기술을 처리 할 수 있도록 쉽게 확장할 수 있음.
    - data-driven한 학습에 있어서 더 많은 데이터가 생겼을 때, 각 요소를 업데이트 하는 것이 용이해짐.

    발화 생성 전략또한 다양한 종류의 speech act를 위한 구성 요소들로 모듈화 되어 있음. 유의미한 대화를 축적하기 위해, 몇몇 생성 모듈은 다양성한 답변을 생성하는 랜덤성이 추가되어 있음.


