---
layout: page
title: About Me
permalink: /about/
---

**당근 마켓**에서 Machine Learning Engineer로 다양한 Machine Learning Application 연구/개발을 하고 있습니다. 자연어 처리, 컴퓨터 비전 등 전반적인 머신러닝/딥러닝에 관심이 많습니다. 블로그에는 주로 공부한 내용(논문 리뷰 등)을 기록합니다.

---

### Education

**DGIST**
Bachelor, Basic of Science *(2015 - Present)*

- 산업기능요원 복무 중입니다.(휴학)

### Work Experience

**Daangn Market**, Machine Learning Engineer *(Jul 2020 - Present)*

**Scatterlab**, Machine Learning Software Engineer *(Mar 2019 - Jun 2020)*

- Research on Pre-training Language Models for Dialogue System. [[Blog post](https://blog.pingpong.us/dialog-bert-pretrain/)]
- Research on [Knowledge Distillation](https://speakerdeck.com/scatterlab/overview-and-recent-research-in-distillation) methods for model compression and BERT serving.
- Develop Multitask Fine-tuning (MT-DNN) system for NLU inference Integration.
- Develop multi-turn reaction and response selection model using BERT. [[Blog post](https://blog.pingpong.us/dialog-bert-multiturn/), [Demo](https://demo.pingpong.us/multi-turn-reaction/)]

**Naver Clova AI**, AI Research Intern *(Dec 2018 - Feb 2019)*

### Research Interest

Natural Language Processing, Computer Vision, Machine Learning, Deep Learning

### Extracurricular Activities

- Little Big Data presenter [[Slide](https://drive.google.com/file/d/0B7WJKAIuHDSeS0h4T2NRem43UG9PcDk3YzVBUkJOTmJWc0NZ/view)] *(Aug 2010)*
- World Friends IT Volunteer, Hanoi, Vietnam *(Jul 2016 – Aug 2016)*

### Awards
- MaaS Hackathon(2019), *3rd prize* – worked as machine learning engineer	
- SKT Blockchain Hackathon(2018), *1st prize* – worked as backend developer
- Crypto On the beach Hackathon(2018), *1st prize* – worked as backend developer

---

### BLOG POST

총 포스트 수: {{ site.posts | size }}개

{% assign number_of_posts = 0 %} {% for post in site.posts %}{% assign currnet_year = post.date | date: "%Y" %}{% assign previous_year = post.previous.date | date: "%Y" %}{% assign number_of_posts = number_of_posts | plus: 1 %}{% if currnet_year != previous_year %}

- {{ currnet_year }}년 : {{ number_of_posts }}개의 포스트{% assign number_of_posts = 0 %}{% endif %}{% endfor %}

**BERT Series Review**: [BERT(NAACL 2019)](https://baekyeongmin.github.io/paper-review/bert-review/), [MT-DNN(arXiv 2019)](https://baekyeongmin.github.io/paper-review/mt-dnn/), [RoBERTa(arXiv 2019)](https://baekyeongmin.github.io/paper-review/mt-dnn/), [ALBERT(ICLR 2020)](https://baekyeongmin.github.io/paper-review/albert-review/), [StructBERT(ICLR 2020)](https://baekyeongmin.github.io/paper-review/structbert-review/), [Transformer-XL(ACL 2019)](https://baekyeongmin.github.io/paper-review/transformer-xl-review/), [XLNet(NeurIPS 2019)](https://baekyeongmin.github.io/paper-review/xlnet-review/), [UniLM(NeurIPS 2019)](https://baekyeongmin.github.io/paper-review/unilm-review/) [SchuBERT(ACL2020)](http://baekyeongmin.github.io/paper-review/schubert-review/)

**Dialogue Modeling(System) Review**: [Transformer for Learning Dialogue(ACL 2020)](https://baekyeongmin.github.io/paper-review/hierarchical-multiparty-transformer/), [Masked Hierarchical Transformer(AAAI 2020)](https://baekyeongmin.github.io/paper-review/masked-hierarchical-transformer-review/), [Graph-Structured Network(IJCAI 2019)](https://baekyeongmin.github.io/paper-review/GSN-review/), [ReCoSa(ACL 2019)](https://baekyeongmin.github.io/paper-review/ReCoSa-review/), [DialogueRNN(AAAI 2019)](https://baekyeongmin.github.io/paper-review/dialogueRNN-review/), [SoundingBoard(Alexa Prize 2017 1st prize)](https://baekyeongmin.github.io/alexa-prize/sounding-board-review/)

**Deep Learning Paper Review**: [Graph Convolutional Network (ICLR 2017)](https://baekyeongmin.github.io/paper-review/gcn-review/)
