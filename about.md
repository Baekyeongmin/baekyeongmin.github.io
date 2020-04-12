---
layout: page
title: About Me
permalink: /about/
---

**Scatterlab**에서 Machine Learning Software Engieer로 자연어 처리(NLP) 연구/개발을 하고 있습니다. 머신러닝, 자연어 처리 전반 중 특히 일상 대화 시스템(Open Domain Dialogue System)에 관심이 많습니다. 블로그에는 주로 공부한 내용(논문 리뷰 등)을 기록합니다.

---

### Eductation

**DGIST**
Bachelor, Basic of Science *(2015 - Present)*

- 산업기능요원 복무 중입니다.(휴학)

### Work Experience

**Scatterlab**, Machine Learning Software Engineer *(Mar 2019 - Present)*

- Research on Pre-training Language Models for Dialogue System. [[Blog post](https://blog.pingpong.us/dialog-bert-pretrain/)]
- Research on [Knowledge Distillation](https://speakerdeck.com/scatterlab/overview-and-recent-research-in-distillation) methods for model compression and BERT serving.
- Develop Multitask Fine-tuning (MT-DNN) system for NLU inference Integration.
- Develop multi-turn reaction and response selection model using BERT. [[Blog post](https://blog.pingpong.us/ml-dialog-bert-multiturn/), [Demo](https://demo.pingpong.us/multi-turn-reaction/)]

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
