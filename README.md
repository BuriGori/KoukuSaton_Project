# 2021 NLP 프로젝트
> 프로젝트 이름 : 쿠크세이튼

> 팀명 : 발탄하드

> 팀원 : 안홍현(팀장), 김영광, 장민수

## 프로젝트 주제
> "이건 도대체 어떤 알고리즘을 사용하는 거야?"

비전공자도 쉽게 Python을 배우고 소규모 프로젝트도 함께 진행할 수 있는 현재 기업들의 입사과정에는 코딩테스트가 필수로 진행되고 있다. 하지만, 코딩테스트를 준비하면서 문제를 푸는 사람들이라면
공통적으로 위와 같은 질문을 해본 적 있을 것이다. 알고리즘을 이해하고 사용할 줄 알면 될줄 알았던 코딩테스트에는 쉽게 어떤 알고리즘을 활용하는지 알려주지 않았기 때문이다. 물론 문제를 보고
해당하는 알고리즘을 생각하고 구현을 할 수 있는 것이 최고의 방법이지만 초보자에겐 이러한 과정이 어려울 수 있기 때문에 문제를 보고 알고리즘을 분류하고자 하는 아이디어에서 시작되었다.

## 목표
1. 알고리즘 문제가 주어지면 Input과 Output이 아닌 문제의 스토리를 보고 알고리즘을 파악해서 어떤 분류에 속하는지 알려준다.
2. 알고리즘 풀이의 대표적인 사이트 백준(BOJ.kr) 문제들을 크롤링 하여 training 및 test set으로 나누어 진행한다.
3. 한글 알고리즘 문제를 중점으로 하여 더 많은 언어의 문제를 분류하는 것이 최종목표이다.

## 한계
* 크롤링
> 목표하였던 백준 사이트 문제를 크롤링하는 것은 문제가 없었으나 알고리즘 분류에 대해서는 로그인된 회원들에게만 제공되므로 알고리즘 분류를 하기 위해서는 자동 로그인을 통해서 크롤링을 진행하려고 했으나
백준사이트에서 자동 로그인 방지가 활성화 되어 다른 사이트를 찾아보게 되었고 알고리즘 문제가 많이 있는 CodeForces라는 사이트를 참고하게 되었다. 사이트 내에서도 아랍어, 러시아어와 같은 언어가 아닌 영어로 된 문제들을 중심으로
알고리즘을 분리하였고 이렇게 크롤링을 진행한 결과 알고리즘 종류가 굉장히 많이 나오게 되었다.

* 알고리즘 분류
> Codeforces에서 제공되는 알고리즘을 분류한 결과 38가지의 알고리즘의 종류가 나왔으며 이를 csv파일로 전환하였을 때, 크롤링시 알고리즘의 이름을 그대로(예: 'binary search', 'greedy', 'math')배열의
형태로 저장하여 학습하기에 적합한 구조가 아니었다. 하지만, 알고리즘 분류를 읽어들이기 쉬운 배열의 형태로([0,0,1,0,1,1,0,0,0,..])과 같은 형태로 전환하면서 학습을 다시 진행할 수 있었고 새로운 데이터를
수집하게 되는 경우 알고리즘 분류에서 신경을 써가면서 수집하므로 해결방안을 찾았다.

* 데이터 양
> Codeforces에서 제공되는 문제의 양이 많을 것으로 예상되었지만 알고리즘 문제중에서 다른 언어를 제외한 영어만 데이터 수집을 하여 최종적으로 800개라는 작은 데이터가 모여서 학습을 하는데 어려움이 되었고
최대한 사용 가능한 결과값을 만들어 내기 위해서 연산의 양이 늘어나는 점을 감수하면서 epoch의 값을 높이기로 결정하였다.

## 환경
* Ubuntu-18.04
* GPU: RTX3090 x 4

## 구현
```python
import requirments
import transformers
import pandas
import torch
import scikit-learn
```

## 마무리
최종적인 결론에 도달하지 못했다는 점에서 아쉬운 점들이 많으나 프로젝트를 진행하는 과정에서의 배움이 있기 때문에 더 도전할 수 있었던 과제였다. 1가지 알고리즘의 종류를 제공하는 것 만으로도 코딩문제를 푸는 것에서는 큰 도움이 되기에 이런 관점에서는 목적에 맞는 동작을 하는 프로그램이지만 완벽히 모든 알고리즘을 제공하는 방법에는 더 많은 데이터와 시간이 필요했다. 아쉽지만 지금까지의 결과물로 프로젝트를 마무리 하려고 한다. 작은 결과에서 큰 결과로 이루어지는 과정을 다음 프로젝트를 하게 된다면 이어 나가고 싶다.

## 참고
* PyTorch 공식 Docs(https://pytorch.org/docs/stable/index.html)
* 밑바닥부터 시작하는 딥러닝 (1)
* 밑바닥부터 시작하는 딥러닝 (2)
