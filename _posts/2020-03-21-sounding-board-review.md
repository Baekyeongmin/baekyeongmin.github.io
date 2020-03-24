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

# Introduction

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


# System Architecture

프론트 엔드, 미들 엔드, 백 엔드로 구성되며, 이 들 또한 각각의 세부 구성요소들로 나눠지는 모듈화된 구조임.

- 프론트 앤드: Sounding Board는 Alexa Skill Kit(ASK)에 포함된 음성 인식(Speech Recognition), 텍스트-음성 변환(TTS)을 통해 유저와 소통함.

- 미들 엔드(AWS Lambda 서비스 이용): 크게 Natural language understanding(NLU), Dialogue management(DM), Natural language generation(NLG) 세개의 주요 시스템 모듈로 구성됨. 각 모듈에서 백 엔드를 이용함.

- 백 엔드: 파싱 서비스를 제공하는 스탠퍼드 CoreNLP, 토픽으로 인덱싱된 컨텐츠가 저장되어 있는 AWS DynamoDB, QA, 조크 서비스를 위한 Evi.com 등으로 구성되며, 미들 엔드와 소통함.

## Natural language understanding

유저의 발화로 부터 화자의 의도 혹은 목적, (잠재적인) 토픽, 감정, 스텐스 등 여러 타입의 정보들을 추출하는 모듈임. 이는 위의 표와 같이 크게 5개의 attribute로 구성됨.

- 입력: ASR hypotheses, VUI output, Dialogue State
  - ASR: Autometic Speech Recognition 모듈
  - VUI: [Voice User Interface](https://developer.amazon.com/en-US/alexa/alexa-skills-kit/vui)
  - Dialogue State

위 입력으로 5개 타입의 정보를 추출함. 이는 후속 모듈들에서 이용됨
- primary intent
  - 유저 발화의 의도를 22개로 구분하는데, 이는 다양한 대화 전략들에서 이용됨.
  - 크게 contents retrieval commands, nevigation commands, common Alexa commands, converse intent의 4종류로 구분
  - contents retrieval commands: popular topics, facts, opinions, jokes, general news, sports news, personality quiz, question answers, unspecified의 9 종류가 있음. 컨텐츠를 요구하는 발화가 들어왔을 때 이 intent로 메핑되는듯
  - navigation commands: human-to-mahcine 대화이기 때문에 필요한 help, repeat, next, cancel 등의 명령
  - standard Alexa command: Alexa가 지원하는 기능에 대한 명령, 8개의 Amazon의 built-in intents "음악 켜줘", "책 읽어줘" 등의 발화에 해당됨. -> Sounding Board는 이에 대한 권한이 없기 때문에, 이러한 command가 감지되면 제약사항과 지침에 대해 응답함.
  - converse intent: 위의 command들이 아닌 나머지 모든 발화들에게 할당하는 intent, 일반적으로 일상대화에서 이용되는 informing, decision/answer, expressing opinions/feelings, asking questions, 등이 해당함. 이 의도들 중 대부분은 이전 턴들을 참조한다는 점에서 컨텍스트 또한 의존관계가 있음.

- question type
  - 다양한 유저 대화를 분석해본 결과, 대부분의 질문은 4가지 유형 중 하나로 매핑된다는 것을 보여주었음. 이들은 각각 DM에서 다른 전략으로 다뤄짐.
  - command: 정중한 방식(질문 형태의) 명령, "우리 Mars mission에 대해 이야기 해볼까?"
  - backstory/ personal question: socialbot의 페르소나에 관한 질문 - 이름, 생일, 취미 등등
  - factual quesion: 사실에 근거한 명확한 답이 있는 질문 "미국의 대통령은 누구야?"
  - question on sensitive topics, advice questions: 성이나 폭력성, 마약 등에 관련된 질문, "내가 어떤 주식을 사야할까?" 등의 어려운 질문 -> socialbot으로써, 대답하기 민감하거나 할 수 없는 질문들에 대한 타입인듯

- candidate topics: 모든 명사구 토픽이 될 수 있음. 문장 내의 모든 명사구를 후보 토픽에 저장하고, 적절하지 못한 토픽("this", "yep", "something" 등)과 민감한 토픽을 제거합니다.

- primary topic: VUI에서 인식된 토픽이 있다면 해당 토픽이 선택됨. 그 외의 경우, 가장 긴 명사구가 선택됨.

- user reaction: socialbot이 컨텐츠를 제공했을 때, 유저들은 가끔 커멘트에서 감정을 표현하거나 긍정 혹은 부정적인 스텐스로 반응할 수 있음. 3개의 리엑션 분류기를 갖고 있음.(각 분류기는 리엑션의 특정 차원에 대해 집중함.)
  - 확인에 대한 사용자의 반응: 승인, 거부, 확실하지 않음, null
  - 의견적인 부분에서 유저의 스텐스: 동의, 동의하지 않음, 확실하지 않음, null
  - fake나 joke에 대한 사용자의 감정: 호/불호, neutral, null
  - null은 classifier의 컨텍스트와 일치하지 않는 경우에 매핑됨, 확인 질문이 아닌경우 1은 Null, 인식기가 오류인 경우 등

## Dialogue management

- 계층적 구조의 state-based 대화 모델을 사용함. state는 이산적인 상호작용 타입, personality quiz의 결과, 이전에 다뤄졌던 컨텐츠에 대한 기억 등임
- 대화를 전체적으로 관리하는 master processing sequence + 대화 시그먼트를 다루는 miniskills(conversation mode)
- 계층적 구조를 통해 새로운 능력을 추가하는 과정이 쉬워지고, 상위 레벨에서의 대화 모드의 잦은 변동을 다루기에 유용함.

- 각각의 턴마다 DM 모듈은 유저의 의도를 다루고, 대화의 토픽에 대한 조건을 만족시키는 대화전략을 확인한다. 마스터 프로세싱 레벨에서 목표는 대화 모드와 적절한 미니스킬을 확인하는 것이다.
- state-independent 단계는 새로운 대화 시그먼트가 명확히 초기화되는 경우(명확한 토픽 요청, 다른 명령의 타입 등)인 경우 진행됨.
- 위 경우가 아닌경우, state-dependent 대화 전략이 실행됨.
- 두 경우 모두 사용자의도/토픽에 대한 제약조건을 만족시키는 miniskill들을 투표함.
- 가장 자세한 토픽에 대응되는 미니스킬이 우선시되지만, 나머지의 경우 랜덤하게 선택됨.(연속된 턴에서 같은 미니스킬의 사용을 피하기위해)
- 미니스킬은 특정 대화모드의 시그맨트를 관리한다. 다른 미니스킬들은 다른 대화 전략을 갖고 있고, 다른 시스템 액션을 취한다.
- 같은 토픽의 지속은 다른 소스로부터의 컨텐츠 제공을 포함한다. 그러나 특정 이야기에 대해 깊이 들어가는 컨텐츠 소스를 위한 구현은 없다.
- state-dependent한 과정은 에러나 대화에서의 문제를 찾고 이에 대응하는 능력이 포함되어 있다. 부정적인 감정, 만족도 등에 대해
- 궁극적으로 DM 모듈은 speech act를 만들고, dialogue state를 업데이트 함.

- Content retrieval: 위의 두 단계 모두 종종 컨텐츠를 찾아야 하는 액션을 포함한다.(컨텐츠 검색에 대한 명령이ㅔ나 새로운 컨텐츠를 제시함으로써, 대화를 이끌어 나가고 싶은 경우) 
  - 컨텐츠 검색 실패를 줄이기 위해, backoff 전략이 사용됨. 먼저 DM 미ㅔ니스킬은 인풋 프레임을 만족하는 컨텐츠를 찾으려고 시도함.(primaryIntent - 컨텐츠 타입, primaryTopic) 이 조건을 만족하는 컨텐츠가 없는경우, 컨텐츠 타입의 제약조건을 제거하고, 토픽에대한 제약조건을 완화한다.(candidateTopic으로) DM 모듈은 컨텐츠가 검색되지 않는 경우, 실패 액션을 시도한다.(이 토픽에 대해 말할 것이 없음을 유저에게 알림)
- Error handling: 두가지 종류의 에러를 핸들링함. (시스템 에러, 이해 에러)  
  - 시스템 에러는 서비스 예외(requenst time-out), 버그로 인한 실패 등을 포함함. 이러한 경우, dialogue state를 초기화하고, 적절한 사과와함께 대화를 재시작하는 부드러원 예외처리를 한다.
  - 이해 에러는 ASR에러, 기대하지 않은 유저의 의도, 언어 처리 에러 등이 해당한다. 시스템은 대화 전략이 요구하는 input frame이 없는 경우 이러한 에러를 감지함. 이러한 에러가 감지된경우, 시스템은 잘못된 이해를 인정하고, 대화를 계속할 것을 유저에게 제안한다. 시스템이 두번째 이해 에러를 감지했을 때, DM모듈은 새로운 토픽 혹은 미니스킬을 선택한다.(계속적으로 바꾸어 말하는 것을 요청하는 것은 랜덤한 토픽 변화보다 화나게 하기 때문에)
- Satisfaction detection: 시스템이 불쾌하거나 모욕적이거나 지루한 컨텐츠를 제공했을 때, 다른 종류의 문제가 발생함. 유저는 토픽 변화를 요청할 수 있지만, 유저 상호작용의 분석은 불만을 표시하는 경우가 많았다. 이런 경우 시스템은 문제를 발견하고 토픽 변화를 초기화해야한다. 이 문제를 해결하기 위해, 토픽을 바꾸어야하는지 판단하는 간단한 이진 분류모델을 만들었다. (이 경우가 더 consistent한 annotation을 받을 수 있었기 때문에) 사용자 만족은 토픽 변화가 필요하지 않았다. 2381개의 데이터를 만들었고, 적은양의 데이터로 인해 n-gram feature를 이용한 logistic regression classifier를 만들었다.
- Speech act selection: 유저의 턴을 다루는 것은 미니스킬에 의존하여 엑션의 시퀀스를 포함한다. 그러나 최종 결과는 speech act를 선택하도록 요구된다. 반응은 여러개의 speech act를 포함할 수 있다. grounding, inform, request, instruction 4가지 종류가 있다.
  - grounding을 위해 DM은 6개의 넓은 종류의 speech act(back-channel, echo of user request for confirmation, 3가지 문제 인정(잘못된 이해, 컨텐츠의 부족, 유저의 도전), 감사)로 분류한다. 이러한 피드백은 중요하고 유저의 발화에 대한 에이전트의 이해를 전달할 수 있다. grounding act는 유저의 발화를 받았음을 알리는 것을 포함한다. 그리고 NLU모듈에 의해 생산된 input frame, 컨텐츠 검색결과, 토픽 변화 감지, 에러 헨들링에 의해 결정된다.
  - inform은 유저에게 컨텐츠가 제공되어야하고, 컨텐츠와 쌍이 이뤄진 경우 사용된다.
  - request는 확인 질문, 토픽에 대한 요청, 사용자가 의견을 제시하는 제안 등이 포함됨.
  - inform act는 주로 request act와 결합되고, 이를 통해 시스템 발화에 커멘트를 제공하도록 장려합니다.
  - 테스트 유저에게 자극을 줄 수 있는 명확한 확인을 최소화합니다.
  - instruction act는 dialogue state나 error detection에 따른 도움 메세지 입니다.

  