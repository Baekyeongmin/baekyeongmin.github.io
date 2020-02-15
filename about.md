---
layout: page
title: About Me
permalink: /about/
---

**Scatterlab**에서 Machine Learning Software Engieer로 자연어 처리(NLP) 연구/개발을 하고 있습니다. 머신러닝, 자연어 처리 전반 중 특히 일상 대화 시스템(Open Domain Dialogue System)에 관심이 많습니다. 블로그에는 주로 공부한 내용(논문 리뷰 등)을 기록합니다.

### Eductation

**DGIST**
Bachelor, Basic of Science *(2015 - Present)*

- 산업기능요원 복무 중입니다.(휴학)

### Work Experience

**Scatterlab**, Machine Learning Software Engineer *(Mar 2019 - Present)*

**Naver Clova AI**, AI Research Intern *(Dec 2018 - Feb 2019)*

---

### BLOG POST

총 포스트 수: {{ site.posts | size }}개

{% assign number_of_posts = 0 %} {% for post in site.posts %}{% assign currnet_year = post.date | date: "%Y" %}{% assign previous_year = post.previous.date | date: "%Y" %}{% assign number_of_posts = number_of_posts | plus: 1 %}{% if currnet_year != previous_year %}

- {{ currnet_year }}년 : {{ number_of_posts }}개의 포스트{% assign number_of_posts = 0 %}{% endif %}{% endfor %}
