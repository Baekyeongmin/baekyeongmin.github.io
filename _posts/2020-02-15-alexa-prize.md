---
title: "Alexa Prize 둘러보기"
layout: post
categories:
  - competition
tags:
  - dialogue
last_modified_at: 2020-02-15T20:53:50-05:00
author: yeongmin
comments: true
---

"사람과 기계가 상호작용하는 방식은 변곡점에 위치해 있고, 대화형 인공지능은 변화의 중심에 있다." Alexa Prize Socialbot Grand Challenge는 매년 Amazon에서 대학(원)생을 대상으로 개최하는 socailbot을 만드는 챌린지입니다. 2017년부터 매년 진행되고 있으며, 현재 2019년 챌린지가 진행되고 있습니다. 이번 포스트에서는 대회가 어떻게 진행되고, 어떤 목표를 달성하고, 평가하는지 살펴보겠습니다.

# Alexa Prize Socialbot Grand Challenge 3(2019)

본 대회는 인기 주제들과 뉴스 이벤트에 대해 **일관되고(coherently)**, 사람을 **사로잡는(engagingly)** 소셜 봇을 만들어 경쟁합니다. 참여 팀들은 Knowledge ecquisition, Natural language understanding(NLU), Natural language generation(NLG), Context modeling, Commonsense reasoning, Dialogue planning 등 다양한 대화형 AI영역을 발전시켜, 목표를 달성합니다.

## 1. 지원 및 참가 팀 선정

아래 그림과 같이 각 팀별로 팀의 비전(해당 팀의 소셜 봇과 사용했을 때 사용자가 느낄 수 있는 느낌), 구축하고자하는 대화 시스템의 전체 구조(기술적 접근), 접근법의 참신함, 영향력, 팀원의 역할, 레퍼런스 등을 기술한 지원서를 제출 합니다. 

![apply](/images/alexa_prize/apply.png){: width="100%"}{: .center}

몇가지 제약사항에 대한 내용도 포함되어 있는데, 모든 봇은 "영어"로 사용자와 상호작용해야 하며, AWS Lamda를 이용해서 호스팅되어야 합니다. 또한 Amazon(후원사)에서 대회에서 사용할 수 있는 소프트웨어, 개발 키트, 라이브러리, API, 문서, 샘플 코드 등을 제공합니다.

지원서와 아래의 평가 항목으로 평가가 진행되고, **10개의 팀**이 선정되어 기술 개발 기간에 참여합니다. 이 팀에 선정되면, 연구비, AWS, Alexa 디바이스 등의 소셜봇 개발에 대한 지원을 받습니다. 

![criteria](/images/alexa_prize/criteria.png){: width="60%"}{: .center}

2019년도 챌린지에서는 아래의 10팀이 참여하고 있습니다. 엄청난 학교들의 경쟁이라, 어떤 결과를 보여줄지 기대가 됩니다 :)

![teams](/images/alexa_prize/teams.png){: width="100%"}{: .center}

## 2. 기술 개발 & 피드백

**Initial Skills Development**: 위의 과정에서 선발된 팀들은 각 팀이 제출한 대로 소셜 봇을 개발합니다. 이 때, 이전 년도 참가 팀들의 결과물들을 이용할 수 있습니다.

**Beta Program**: 실제 Alexa 유저들이 사용할 수 있는지를 검증(+ Amazon의 피드백)하는 과정을 거쳐 중간 결과물 Beta Program을 만듭니다. Amazon은 특정 기간동안 이 프로그램을 내부적으로 평가하고, Interaction Rating, 공격적인 컨텐츠 필터링하는 기능, 가동 시간 등의 기준(실제 사용자들이 상호작용할 수 있는 기준)을 충족하도록 합니다.

- Interaction Ratings: Alexa 사용자들이 각 소셜 봇과 특정 주제(야구 플레이 오프, 유명인, 과학적 혁신 등의 주제)에 대해 대화(상호 작용)하고 평가하는 만족도.

**Feedback**: 위의 기준을 충족한 소셜 봇들에 대해 일정 기간동안 Alexa 실제 사용자들에게 서비스를 제공하고, Interaction Rating을 받습니다. 이 때 수집된 점수들은 각 팀에게 전달됩니다. 해당 기간동안 지속적으로 개발을 진행할 수 있습니다.

**Quarterfinal**: 모든 참가팀을 두 개의 그룹으로 나누고, Alexa 사용자들은 일정 기간동안 각 팀의 소셜봇을 이용하고, 평가합니다. 이를 통해 각 그룹별로 평균 Interaction Rating이 높은 3개의 팀과 Amazon이 여러 기준(Amazon과 관련성, 일관성, 흥미, 속도, 기술적 장점)을 통해 뽑은 1~2개의 팀이 준결승 피드백기간으로 넘어갑니다. (이 외의 팀들도 계속 개발을 진행할 수 있습니다.)

**Semifinal**: 준준결승과 유사한 방식으로 일정기간 평가를 진행하고, Interaction Rating 상위 3개의 팀(준준결승에서 선택된 팀들 중), Amazon이 선정한 2개의 팀(모든 팀 중)이 결승에 진출합니다.

**Technical Publication**: 모든 팀은 소셜 봇을 만들기 위해 진행했던 기술적 접근법, 실험 결과 등을 기술 보고서 형식으로 제출해야 합니다. (2017, 2018년도 참가팀들의 보고서도 공개되어 있습니다.)

결승 이전의 모든 기간동안 각 팀은 피드백(Interaction Rating)을 받으면서 모델을 개선시킬 수 있습니다.

## 3. 결승 및 시상

Amazon은 두 집합의 평가자들을 구성합니다.

- Interactors: 특정 인기 있는 주제로 소셜봇과 턴-by-턴으로 대화(상호작용)하는 사람들
- Judges: 소셜봇과 Interactor들의 상호작용을 평가하는 사람들

결승에 참가하는 각각의 소셜 봇은 동일한 수의 Interactor들과 상호작용 합니다. 대화가 모두 끝난 후 Judge들이 관련성, 일관성, 흥미, 등의 요소에 따라 해당 대화를 1~5등급으로 평가합니다.(경우에 따라 청중 Judge의 평가도 이용할 수 있습니다.)

최종 점수는 평균 최종 등급, 준결승, 베타 프로그램의 등급, 관객 심사위원이 제공한 등금 등이 복합적으로 고려되어 계산됩니다.(동점의 경우 대화의 평균 지속시간을 고려합니다.) 이 점수에 따라 순위를 메깁니다.

1등에게는 $500,000 U.S 달러, 2등에게는 $100,000 U.S 달러, 3등에게는 $50,000 U.S 달러가 주어지며, 1등의 평균 복합 점수가 4.0 이상이고, 20분 이상 지속된 대화가 전체의 2/3이상인 경우 $1,000,000 U.S 달러가 추가적으로 지급됩니다.(Grand Prize)

# 마무리

Alexa Prize는 대화형 AI에 관련된 가장 큰 규모의 챌린지 입니다. 모든 규칙들을 상세히 읽으면서, 좋은 대화 시스템을 만들기 위해 어떤 요소들을 고려해야할지 다시 한 번 생각해볼 수 있었습니다. 최근에는 대화 시스템이 각 상황에 알맞은 응답을 하는 것은 기본이고, 이를 넘어서 얼마나 흥미롭고 매력적인지를 고려하는 것도 큰 비중을 차지하는 것 같다는 느낌을 받았습니다. 특히 Grand Prize의 20분 이상 대화하는 조건은 2년 동안 달성한 팀이 없고, 실제로 사람과 대화하는 느낌을 재현해야 충족시킬 수 있다고 생각합니다. 이번 년도에는 각 팀들이 어떤 방식을 시도 했고, 결과는 어떨지 기대가 됩니다. 한편, 자신이 만든 소셜봇을 아마존 서비스를 통해 실 사용자들의 피드백을 받고 문제를 파악할 수 있는 점에서 엄청난 메리트를 갖는 챌린지 인 것 같습니다. 

다음 포스트에서는 2017년 우승팀 Sounding Board(University of Washington), 2018년 우승팀 Gunrock(University of California, Davis)의 방법론들을 살펴보려고 합니다.

# Reference

- [Alexa Prize Official Website](https://developer.amazon.com/alexaprize)
