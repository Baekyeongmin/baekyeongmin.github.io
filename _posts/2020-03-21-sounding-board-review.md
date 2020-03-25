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

이전 글에서 Alexa Prize에 대해 간단하게 살펴봤는데, 이 글에서는 Socialbot challenge 2017의 우승작 [Sounding Board – University of Washington’s Alexa Prize Submission](https://m.media-amazon.com/images/G/01/mobile-apps/dex/alexa/alexaprize/assets/pdf/2017/Soundingboard.pdf)을 리뷰해보려고 합니다. 이전의 open-domain 대화 시스템은 주어진 컨텍스트에 대해 "적절한", "말이 되는" 답을 하는 것을 목적으로 하는 *chit-chat*에 가까웠습니다. 하지만 Alexa Przie는 소셜봇이 유저가 선택한 특정 주제에 대해 대화를 진행하는 챌린지 이기 때문에, Sounding Board 팀은 컨텐츠(정보)와 유저의 흥미를 집중적으로 고려하여 대답을 생성하는 *task-oriented* 문제로 다루었습니다. 특히 워싱턴대 팀은 본 대회 이전에 만들어 두었던 대화 시스템이 없었고 뉴럴넷을 학습시킬 데이터도 없었기 때문에, 초기 시스템은 간단한 rule-base로 구성되어 밑바닥 부터 시작했습니다. 실 유저와의 대화를 경험하고 이를 통해 데이터를 축적함으로써, 구조는 복잡해졌고 각 구성요소는 머신러닝을 사용하는 방향으로 발전했습니다. 논문에서는 최종 제출한 결과물을 다룹니다.

# 1. Design Philosophy

Sounding Board의 디자인 철학은 대화 전략 및 시스템 엔지니어링에 반영되어 있습니다. 대회를 진행하면서 얻을 수 있는 사용자 피드백은 시스템 구조 결정에 있어서 큰 영향을 미쳤습니다. 대화 전략은 다음과 같이 두 개의 특징을 갖습니다.

- `contents driven`: 유저가 알지 못하거나 들어보지 못했을 관점의 정보를 제공하고, 이를 통해 유저를 사로잡고자 했습니다. 따라서 정보 검색(Information retrieval)의 중요성이 높고, 다양한 범위의 토픽, 유저의 흥미를 다룰 수 있도록 여러 개의 정보 소스를 이용했습니다. *chit-chat*은 일부 포함되어 있으나 주로 대화를 부드럽게 진행시키는 역할을 수행합니다.
- `user driven`: 유저 발화의 감정, 태도, 의도 등을 다차원 표현으로 나타내고, 성격 퀴즈, 불만 감지 등을 통해 사용자의 상태를 지속적으로 인식하고, 대화를 진행하고자 했습니다.

시스템 엔지니어링 전략은 여러개의 컨텐츠 소스들을 적절히 이용하고자 했던 점, 학습을 위한 적절한 대화 데이터가 부족한점이 주로 고려되었습니다. (따라서 end-to-end generation 시스템 등은 시도하지 않았음) 이에 따라 계층적이고 모듈화된 구조를 통해 대화 관리를 하는 전략을 선택했습니다. 이러한 구조 덕분에 새로운 능력을 추가하고 기존의 기능을 업데이트하는 것이 쉬워 확장성이 커졌습니다. 발화 생성 또한 모듈화 되어있어, 여러 모듈들이 speech act에 따라 반응을 생성합니다. 이 때, 몇몇 생성 모듈에는 여러가지 변형들 중 답변 형식을 선택하는 랜덤성을 부여하여 대화의 단조로움을 피하고, 유의미한 대화를 축적하도록 의도했습니다.

# 2. System Architecture

![architecture](/images/sounding_board/architecture.png){: width="100%"}{: .center}

위 그림과 같이 프론트 엔드, 미들 엔드, 백 엔드로 구성되며, 이들 또한 각각의 세부 구성 요소들로 나눠지는 모듈화된 구조입니다.

- 프론트 앤드: 유저와 소통하는 부분으로, Sounding Board는 Alexa Skill Kit(ASK)에 포함된 음성 인식(Automatic Speech Recognition - ASR), 텍스트-음성 변환(Text-to-Speech TTS)을 이용합니다.

- 미들 엔드(AWS Lambda 서비스 이용): 다양한 대화 전략을 가지고 유저의 발화에 대한 반응을 생성하는 부분으로, 크게 Natural language understanding(NLU), Dialogue management(DM), Natural language generation(NLG) 3개의 주요 시스템 모듈로 구성됩니다. 각 모듈에서 필요시 백앤드와 통신합니다.

- 백 엔드: 각 대화 전략에서 이용할 수 있는 데이터와 정보를 제공하는 부분으로, 파싱 서비스를 제공하는 스탠퍼드 CoreNLP, 토픽으로 인덱싱된 컨텐츠가 저장되어 있는 AWS DynamoDB, QA, 조크 서비스를 위한 Evi.com 등으로 구성됩니다.

대화가 진행되는 과정은 다음과 같습니다.
1. 유저가 말을 하면 ASR의 인식 내용, [Voice User Interface](https://developer.amazon.com/en-US/alexa/alexa-skills-kit/vui)의 결과를 얻습니다.
2. 이 정보들과 현재의 dialogue state를 바탕으로 NLU가 DM의 입력으로 이용될 `입력 프레임`을 만듭니다.
3. DM은 이 프레임을 이용하여 대화 전략을 실행하여 Speech act와 컨텐츠를 결정하고, dialogue state를 업데이트 합니다.
4. NLG모듈은 DM의 결과물들을 이용해 반응을 생성하고, ASK에 반환합니다.

## 2.1. Natural language understanding

![input_frame](/images/sounding_board/input_frame.png){: width="80%"}{: .center}

프론트에서 제공되는 정보들로 부터 화자의 **발화를 이해**하고 의도 혹은 목적, (잠재적인) 토픽, 감정, 스텐스 등 여러 타입의 정보들을 추출하여 `input frame`으로 만드는 모듈입니다. 이는 위의 표와 같이 크게 5개의 attribute로 구성됩니다.

문어체에서는 [Ganeral Language Understanding Evaluation(GLUE)](https://gluebenchmark.com) 등 자연어 이해 관련 테스크들이 비교적 잘 정리되어 있지만, 대화체에서는 여전히 정의되지 않은 테스크들이 많고 분산되어 있는것 같습니다. 본 논문에서도 아래와 같이 주어진 환경에 맞춰 여러 NLU 피쳐들을 정의하고 이를 통해 발화를 표현하고자 했습니다.

**입력**: 프론트 엔드의 ASK로 부터 얻을 수 있는 정보들입니다.
1. **`ASR hypotheses`**: Autometic Speech Recognition 모듈의 결과
2. **`VUI output`**: [Voice User Interface](https://developer.amazon.com/en-US/alexa/alexa-skills-kit/vui) 의 결과
3. **`Dialogue State`**: 현재 까지 저장된 대화의 상태(대화가 진행됨에 따라 DM에서 지속적으로 업데이트 됨.)

**출력**: 위 입력으로 5개 타입의 정보를 추출하여 `input frame`을 만듭니다.

1. **`primaryIntent`**
  - DM의 대화 전략을 결정하는데 사용하기 위해, 유저 발화의 의도를 22개로 구분합니다.
  - 크게 `contents retrieval commands`, `nevigation commands`, `common Alexa commands`, `converse intent`의 4종류로 구분됩니다.
  - `contents retrieval commands`: `popular topics`, `facts`, `opinions`, `jokes`, `general news`, `sports news`, `personality quiz`, `question answers`, `unspecified`의 9종류가 있습니다. 유저의 발화가 특정 컨텐츠를 요구할 때 컨텐츠의 종류에 따라 이 중 하나로 매핑됩니다.
  - `navigation commands`: 인간과 봇의 대화이기 때문에, 사용자가 봇에게 명령을 내릴 수 있는데,`help`, `repeat`, `next`, `cancel` 등이 해당됩니다.
  - `standard Alexa command`: Alexa가 기본적으로 지원하는 기능에 대한 명령입니다. 8개의 Amazon의 built-in 의도들("음악 켜줘", "책 읽어줘" 등의 발화)에 해당됩니다. Sounding Board는 이에 대한 권한이 없기 때문에, 이러한 커멘드가 감지되면 제약사항과 지침에 대해 응답합니다.
  - `converse intent`: 위의 커멘드들이 아닌 나머지 모든 발화들에게 할당하는 의도입니다. 일반적으로 일상대화에서 이용되는 `informing`, `decision/answer`, `expressing opinions/feelings`, `asking questions`, 등이 해당합니다. 이 의도들 중 대부분은 이전 턴들을 참조해야 하기 때문에, 컨텍스트 또한 의존관계가 있습니다.

2. **`questionType`**
  - 유저 발화의 질문 종류입니다. 저자들은 다양한 유저 대화 분석을 통해, 대부분을 질문은 4가지 유형 중 하나로 대응시킬 수 있다는 결론을 내렸습니다. 이들은 DM에서 각각 다른 전략으로 다뤄집니다.
  - `command`: 정중한 방식(질문 형태의) 명령입니다. - "우리 Mars mission에 대해 이야기 해볼까?"
  - `backstory/ personal question`: socialbot의 페르소나에 관한 질문입니다. - 이름, 생일, 취미 등등
  - `factual quesion`: 사실에 근거한 명확한 답이 있는 질문입니다. - "미국의 대통령은 누구야?"
  - `question on sensitive topics, advice questions`: socialbot이 대답하기 민감하거나 할 수 없는 성이나 폭력성, 마약 등에 관련된 질문들과 "내가 어떤 주식을 사야할까?" 등의 어려운 질문입니다.

3. **`candidateTopics`**: 토픽의 후보들입니다. 모든 명사구가 토픽이 될 수 있다고 가정하고, 파싱 결과에서 모든 명사구를 저장합니다. 그 후에 적절하지 못한 토픽("this", "yep", "something" 등)과 민감한 토픽을 제거합니다. 토픽의 다양화를 위해 `primaryTopic` 에 대한 컨텐츠 부족 등의 문제에 대응하기 위해 후보들을 유지합니다.

4. **`primaryTopic`**: 대화의 주 토픽입니다. `VUI`에서 인식된 토픽이 있다면 해당 토픽이 선택됩니다. 그 외의 경우, 가장 긴 명사구가 선택됩니다.
  - *긴 명사구일 수록 구체적인 토픽일 것이고, 주 토픽일 가능성이 높을 수 있지만 각 명사구의 의미와 관계 없이 길이로만 결정하기 때문에 나이브한 접근인 것 같습니다. 그러나 2017년이라 만족스러운 기술이 없었거나, 이 방법으로도 충분히 잘되었기 때문에 이 방법을 이용했을 수도 있을 것 같습니다.*

5. **`userReaction`**: 소셜봇이 컨텐츠를 제공했을 때, 유저들은 이에 대한 답변(리엑션)에서 감정을 표현하거나 긍정 혹은 부정적인 스텐스로 반응할 수 있습니다. 리엑션을 3개의 차원으로 나누가 각 차원의 정도를 분류하는 3개의 리엑션 분류기를 만들었습니다.
  - 확인에 대한 사용자의 반응: 승인, 거부, 확실하지 않음, `null`
  - 의견적인 부분에서 유저의 스텐스: 동의, 동의하지 않음, 확실하지 않음, `null`
  - fake나 joke에 대한 사용자의 감정: 호/불호, neutral, `null`
  - `null`은 classifier의 컨텍스트와 일치하지 않는 경우에 매핑됩니다. - 확인 질문이 아닌경우 첫번째 분류기는 `null`, 인식기가 오류인 경우 등

## 2.2. Dialogue management

DM은 계층적 구조의 state-based 대화 모델입니다. 여기서 state는 상호작용 타입, personality quiz의 결과, 이전에 다뤄졌던 컨텐츠에 대한 기억 등 이산적인 요소들의 집합입니다. DM은 대화를 전체적으로 관리하는 `master processing`와 대화의 각 시그먼트를 다루는 `miniskill(대화 모드)`로 구분됩니다. 이러한 계층적 구조를 통해 새로운 능력을 추가하는 과정이 쉬워지고, 상위 레벨에서 대화 모드의 잦은 변동을 다루기에 유용합니다.

각각의 턴에서 DM은 `input frame`으로 부터 전달받은 유저 의도와 토픽에 대한 조건을 만족시키는 대화 전략을 확인합니다. `master processing`에서는 조건을 만족시키는 미니스킬을 확인합니다.
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

# Natural Language Generation

speech act과 컨텐츠를 입력으로 받아 반응을 생성한다. 반응은 4개의 큰 종류 중 최대 세개의 speech act를 포함할 수 있다. Amazon TTS API의 요구사항대로, 반응은 message와 reprompt로 구성됨. 장치는 항상 message를 읽음, reprompt는 장치가 주어진 기간동안 아무것도 듣지 못했을 때, 옵셔널하게 사용됨. grounding act는 주로 반응의 시작에 위치하고, instruction은 repromt에 위치함.

grounding act는 특정 카테고리와 관련된 구절/문장들의 모음들 중 랜덤으로 선택된다. back-channeling("I see", "Cool"), user request echoing ("Looks like you want to talk about news"), misunderstanding apology (e.g., “Sorry, I’m having trouble understanding what you said.”), unanswerable user follow-up questions (e.g. “I’m sorry. I don’t remember the details."), and gratitude (e.g., “I’m happy you like it.”).

inform act는 시작 구절과 DM에 의해 제공된 컨텐츠를 결합한 간단한 템플릿에 의해 구성된다.

request act는 유저의 입력을 요청하는 slot-level의 변형의 형태로 구성된다.

instruction act는 변형이 적은 상황에 맞는 도움말 메시지 모음으로 구성된다.

발음을 보다 정확하게 전달하기 위해 ASK SSML을 활용하여 발음함.

마지막으로 욕설 단어/구절을 비 공격적인 단어로 대체하는 발화 정화기를 통과함.

# Miniskills

Sounding Board는 몇몇의 다른 miniskill들을 갖고 있는데 이들은 dialogue state를 관리하고, 대화 시그먼트의 일관성을 책임진다.

## content-oriented miniskills

컨텐츠 수집과 관리는 성공적인 content-oriented miniskill을 구현하기 위해 중요한 두 단계입니다. 여러 소스(Amazon이 제공하는 트렌딩 토픽, Reddit으로 부터 생성된 컨텐츠, 뉴스 기사, QA, joke)로부터 컨텐츠를 얻음.

몇몇의 다른 miniskill들을 구현함.(트랜딩 토픽, 사실, 의견, 일반적인 뉴스, 스포츠 뉴스, 조크, QA) 특정 소스는 넓은 관심사의 뉴스 혹은 논평을 제공하고 스타일은 대화에 적절하기 때문에 선택됨. 개별적인 대화는 상대적으로 짧아야 하므로, 정보력이 있고, 이해기에 적은 컨택스트를 요구하는 정보들을 쉽게 추출할 수 있는 소스를 선택했음. 욕설, 불쾌한 주제를 필터링

- Tranding topics: 유저가 선택할 수 있는 토픽을 추천함. Amazon에 의해 제공됨.
- Facts: TodayILearned 라는 subreddit에는 흥미로운 사실들이 많이 올라옴 이 포스트의 제목들에 등장하는 토픽들로 인덱싱을 진행했음.
- Opinions: ChangeMyView subreddit을 크롤링했음. 여기의 포스트들은 대부분 논쟁의 여지가 있는 것들임. 의견은 논란의 여지가 있기 때문에, ~~한 레딧의 게시물에 대해 어떻게 생각하는지 궁금합니다 와 같은 방식을 이용함.
- General News: UpliftingNews subreddit에서 검색을 진행하는데, 더 긍정적인 컨텐츠를 담는 것을 기대할 수 있음. 뉴스 제목은 위의 두개에 비해 정보량이 적으나 본문은 매우 김. 따라서 처음에 뉴스 타이틀을 읽어주고 4문장까지 요약문을 읽어줌. 요약문은 Gensim의 TextRank 알고리즘을 이용함.
- Sports News: Sports는 많은 사람들의 삶에서 큰 부분을 차지하고 있고, 사용자들은 소셜봇이 최근의 발전에 대해 의미있게 이야기하기를 원함. 이는 시간에 민감한 주제이므로, 최근 스포츠 이벤트를 긁음. 하지만 이 기사는 너무 긴 문제가 있는데, Washington Posrt 기사에 제공되는 "blurb"와 헤드라인을 결합하면 자세한 정보를 제공하지 않고도, 훌륭한 커버리지를 보여주었음. 컨텍스트가 누락되어 있는 발췌문을 필터링할 수 있는 필터를 개발했음
- Jokes: Evi로 부터 joke를 요구함.
- QA: Evi의 QA엔진으로 연결함. Evi의 컨텐츠가 토픽과 관련이 있는지에 따라 다음 미니스킬을 제안함.(이 부분이 유저 경험을 향상시킴, 특히 Evi가 답을 제공하는데 실패한 경우) Alexa의 이름, 생일, 민감한 토픽, 충고등을 요구할 때에는 피하는 전략으로 다뤄짐. 트랜딩 토픽을 다음 미니스킬로 제공함.

## Personality Assessment

사용자들을 여러 다른 타입으로 유지하는 것은 이점이 있었다. 몇개의 짧은 질문을 통해 사람을 4가지 성격 사분면에 위치시켰다. 외향성과 솔직함을 축으로 이용했음. 외향성은 얼마나 수다스러운지와 사교적인지를 판단하고 수용성은 지적, 예술적으로 호기심을 나타내는 것과 관련이 있음.

5개의 질문 후에 그들의 성격 결과를 얻을 수 있는 옵션을 제공함. 중간중간에 대화를 지속하기 위한 "goofy" 질문을 끼워넣음. 각 질문에 대한 대답을 얻었을 때, 다음 질문을 하기 전에 반응을 제공함. (유저 피드백에서 이 반응이 유저 경험에서 큰 향상을 가져옴)

각 성격 질문은 각 차원에 긍정적/부정적으로 로드되며 사용자를 사분면 상에 매치 시킴. 그 후에 사용자를 디즈니 캐릭테어 할당함. 성격 질문에 답한 사용자는 각 사분면의 유형이 관심을 가질만한 것에 기초하여 토픽을 얻음.

## General miniskills

인사말: 오프닝에서 사용되며, 대화에서 한번만 사용됨. how-are-you 질문으로 대화를 초기화하고, 사용자의 대답에 따라 공감함.
메뉴: 사운딩보드의 기능을 사용자에게 소개함. 시스템이 대화를 진행시키기 위해 다음 컨텐츠 지향 미니 스킬을 결정할 수 없는 경우 이용함.
종료: 사운딩보드를 종료하기 위해 명시적으로 중지를 말한 유저를 지시함. 발동 조건에 따라 사운딩 보드는 다른 grounding speech act를 instruction전에 더함. Alexa command에 의해 이슈가 발생되었을 때, 시스템의 한계를 설명함. stop command에 의해 발동되었을 때, 채팅에 대한 감사가 추가됨.

