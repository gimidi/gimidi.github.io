```python
import pandas as pd
from glob import glob
```


```python
import os
```


```python
os.listdir('data/')[1:]
```




    ['3_문어체_뉴스(1).xlsx',
     '2_대화체.xlsx',
     '1_구어체(1).xlsx',
     '3_문어체_뉴스(3).xlsx',
     '3_문어체_뉴스(4).xlsx',
     '1_구어체(2).xlsx',
     '4_문어체_한국문화.xlsx',
     '3_문어체_뉴스(2).xlsx']




```python
df = pd.DataFrame(columns = ['원문','번역문'])
path = 'data/'

file_list = os.listdir('data/')[1:]

for data in file_list:
    temp = pd.read_excel(path+data, engine = 'openpyxl')
    df = pd.concat([df,temp[['원문','번역문']]])
```

    /usr/local/lib/python3.8/dist-packages/openpyxl/styles/stylesheet.py:221: UserWarning: Workbook contains no default style, apply openpyxl's default
      warn("Workbook contains no default style, apply openpyxl's default")



```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>원문</th>
      <th>번역문</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>스키너가 말한 보상은 대부분 눈으로 볼 수 있는 현물이다.</td>
      <td>Skinner's reward is mostly eye-watering.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>심지어 어떤 문제가 발생할 건지도 어느 정도 예측이 가능하다.</td>
      <td>Even some problems can be predicted.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>오직 하나님만이 그 이유를 제대로 알 수 있을 겁니다.</td>
      <td>Only God will exactly know why.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>중국의 논쟁을 보며 간과해선 안 될 게 기업들의 고충이다.</td>
      <td>Businesses should not overlook China's dispute.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>박자가 느린 노래는 오랜 시간이 지나 뜨는 경우가 있다.</td>
      <td>Slow-beating songs often float over time.</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>200536</th>
      <td>이곳은 수목과 높이 6∼8m 아파트 옹벽 등으로 야간보행이 취약하고, 노후한 공공시...</td>
      <td>Students that attend the Borame Elementary Sch...</td>
    </tr>
    <tr>
      <th>200537</th>
      <td>박원갑 국민은행 더블유엠(WM)스타자문단 수석부동산전문위원은 “광역급행철도(GTX)...</td>
      <td>Park Won-gap, a senior real estate expert at W...</td>
    </tr>
    <tr>
      <th>200538</th>
      <td>2007년 3월 4일에는 정월대보름맞이 전통문화 세시풍속 유지전승 국악한마당이란 이...</td>
      <td>On March 4, 2007, the festival was held at the...</td>
    </tr>
    <tr>
      <th>200539</th>
      <td>고창 지역의 문화 축제로는 고창청보리밭축제, 고창모양성제, 고창복분자축제, 고창수박...</td>
      <td>There are cultural festivals in Gochang such a...</td>
    </tr>
    <tr>
      <th>200540</th>
      <td>포상 대상은 ‘모범 중소기업 대표’(제조분야, 유통·서비스분야) ‘모범중소기업 근로...</td>
      <td>There are 4 parts to reward such as Representa...</td>
    </tr>
  </tbody>
</table>
<p>1402033 rows × 2 columns</p>
</div>




```python
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time
```


```python
from torchtext import data 
from konlpy.tag import Okt

tokenizer = Okt()
```


```python
data.
```




    <module 'torchtext.data' from '/usr/local/lib/python3.8/dist-packages/torchtext/data/__init__.py'>




```python
def tokenize_kor(text):
    """한국어를 형태소 기준으로 tokenizer해서 단어들을 리스트로 만듦"""
    return [text_ for text_ in tokenizer.morphs(text)]

def tokenize_en(text):
    """영어를 split tokenizer해서 단어들을 리스트로 만듦"""
    return [text_ for text_ in text.split()]

# 필드 정의
SRC =data.Field(tokenize = tokenize_kor,
                init_token = '<sos>',
                eos_token = '<eos>',batch_first = True,lower = True)

TRG =data.Field(tokenize = tokenize_en,
                init_token = '<sos>',
                eos_token = '<eos>',batch_first = True,
                lower = True)
```


```python
df.sample(frac=1)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>원문</th>
      <th>번역문</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>스키너가 말한 보상은 대부분 눈으로 볼 수 있는 현물이다.</td>
      <td>Skinner's reward is mostly eye-watering.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>심지어 어떤 문제가 발생할 건지도 어느 정도 예측이 가능하다.</td>
      <td>Even some problems can be predicted.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>오직 하나님만이 그 이유를 제대로 알 수 있을 겁니다.</td>
      <td>Only God will exactly know why.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>중국의 논쟁을 보며 간과해선 안 될 게 기업들의 고충이다.</td>
      <td>Businesses should not overlook China's dispute.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>박자가 느린 노래는 오랜 시간이 지나 뜨는 경우가 있다.</td>
      <td>Slow-beating songs often float over time.</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>200536</th>
      <td>이곳은 수목과 높이 6∼8m 아파트 옹벽 등으로 야간보행이 취약하고, 노후한 공공시...</td>
      <td>Students that attend the Borame Elementary Sch...</td>
    </tr>
    <tr>
      <th>200537</th>
      <td>박원갑 국민은행 더블유엠(WM)스타자문단 수석부동산전문위원은 “광역급행철도(GTX)...</td>
      <td>Park Won-gap, a senior real estate expert at W...</td>
    </tr>
    <tr>
      <th>200538</th>
      <td>2007년 3월 4일에는 정월대보름맞이 전통문화 세시풍속 유지전승 국악한마당이란 이...</td>
      <td>On March 4, 2007, the festival was held at the...</td>
    </tr>
    <tr>
      <th>200539</th>
      <td>고창 지역의 문화 축제로는 고창청보리밭축제, 고창모양성제, 고창복분자축제, 고창수박...</td>
      <td>There are cultural festivals in Gochang such a...</td>
    </tr>
    <tr>
      <th>200540</th>
      <td>포상 대상은 ‘모범 중소기업 대표’(제조분야, 유통·서비스분야) ‘모범중소기업 근로...</td>
      <td>There are 4 parts to reward such as Representa...</td>
    </tr>
  </tbody>
</table>
<p>1402033 rows × 2 columns</p>
</div>




```python
10000/1402033
```




    0.007132499734314385




```python
df_shuffled=df.sample(frac=0.01).reset_index(drop=True)
```


```python
df_=df_shuffled.iloc[:10000]
```


```python
df_.shape

```

글을 쓸려면 이렇게 정리를 할 수 있습니다. 설명을 넣으려면 이렇게 설명을 넣을수있습니다.

 이렇게 설명하면 되는거네?

글씨체가 왤케큰거죠..?


    (10000, 2)




```python
from sklearn.model_selection import KFold
```


```python
kf=KFold(n_splits=5,shuffle=True,random_state=6111)
```


```python
for train_idx, valid_idx in kf.split(df_['원문']):
    print(train_idx,valid_idx)
    # 얘네끼리 안겹칠줄 알았는데 겹치네?
    
    trn = df_.iloc[train_idx]
    val = df_.iloc[valid_idx]
```

    [   1    2    3 ... 9997 9998 9999] [   0    5   11 ... 9990 9995 9996]
    [   0    1    2 ... 9997 9998 9999] [   6    7   15 ... 9989 9993 9994]
    [   0    1    3 ... 9995 9996 9997] [   2    8   12 ... 9992 9998 9999]
    [   0    1    2 ... 9997 9998 9999] [   4   10   13 ... 9962 9979 9980]
    [   0    2    4 ... 9996 9998 9999] [   1    3    9 ... 9986 9991 9997]



```python
len(trn), len(val)
```




    (8000, 2000)




```python
path='minidi/'
```


```python
trn.to_csv(path + 'trn.csv',index = False)
val.to_csv(path + 'val.csv',index = False)
```


```python
from torchtext.data import TabularDataset
```


```python
train_data, validation_data =TabularDataset.splits(
    path=path, train='trn.csv',validation='val.csv',format='csv',
    fields=[('원문', SRC), ('번역문', TRG)], skip_header=True)
```


```python
SRC
```




    <torchtext.data.field.Field at 0x7f73f1434f10>




```python
print(vars(train_data.examples[30])['원문'])
```

    ['특히', '아프리카', '콩고민주공화국', '(', '민주', '콩고', ')', '에서', '내전', '중', '성폭행', '을', '당한', '여성', '수만', '명', '을', '치료', '한', '산부인과', '의사', '드니', '무케게', '가', '2016년', '서울', '평화상', '수상자', '다', '.']



```python
print(vars(train_data.examples[30])['번역문'])
```

    ['in', 'particular,', 'the', 'gynecologist', 'denis', 'mukege,', 'who', 'treated', 'tens', 'of', 'thousands', 'of', 'women', 'who', 'were', 'sexually', 'assaulted', 'during', 'the', 'civil', 'war', 'in', 'the', 'democratic', 'republic', 'of', 'congo', '(democratic', 'congo),', 'is', 'the', 'winner', 'of', 'the', '2016', 'seoul', 'peace', 'prize.']



```python
SRC.build_vocab(train_data, min_freq = 2)
```


```python
vars(SRC.vocab)
```




​    




```python
TRG.build_vocab(train_data, min_freq = 2)
```


```python
vars(TRG.vocab)
```




​    




```python
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
BATCH_SIZE = 128
```

    cuda:1



```python
vars(train_data[30])
```




    {'원문': ['특히',
      '아프리카',
      '콩고민주공화국',
      '(',
      '민주',
      '콩고',
      ')',
      '에서',
      '내전',
      '중',
      '성폭행',
      '을',
      '당한',
      '여성',
      '수만',
      '명',
      '을',
      '치료',
      '한',
      '산부인과',
      '의사',
      '드니',
      '무케게',
      '가',
      '2016년',
      '서울',
      '평화상',
      '수상자',
      '다',
      '.'],
     '번역문': ['in',
      'particular,',
      'the',
      'gynecologist',
      'denis',
      'mukege,',
      'who',
      'treated',
      'tens',
      'of',
      'thousands',
      'of',
      'women',
      'who',
      'were',
      'sexually',
      'assaulted',
      'during',
      'the',
      'civil',
      'war',
      'in',
      'the',
      'democratic',
      'republic',
      'of',
      'congo',
      '(democratic',
      'congo),',
      'is',
      'the',
      'winner',
      'of',
      'the',
      '2016',
      'seoul',
      'peace',
      'prize.']}




```python
from torchtext.data import Iterator
train_iterator = Iterator(dataset = train_data, batch_size = BATCH_SIZE)
valid_iterator = Iterator(dataset = validation_data, batch_size = BATCH_SIZE)
```


```python
train_iterator, test_iterator = Iterator.splits(
    (train_data, validation_data),
    batch_size=BATCH_SIZE,
    shuffle=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
```


```python
for batch in train_iterator:
    print(vars(batch))
#     print(batch.text)
#     print(batch.label)
    break
```

    {'batch_size': 128, 'dataset': <torchtext.data.dataset.TabularDataset object at 0x7f73d4efe100>, 'fields': dict_keys(['원문', '번역문']), 'input_fields': ['원문', '번역문'], 'target_fields': [], '원문': tensor([[   2,  900,  914,  ...,    1,    1,    1],
            [   2,  149,  334,  ...,    1,    1,    1],
            [   2,    5,  497,  ...,    1,    1,    1],
            ...,
            [   2,  247,   12,  ...,    1,    1,    1],
            [   2,    0, 5848,  ...,    1,    1,    1],
            [   2,  817,   15,  ...,    1,    1,    1]], device='cuda:0'), '번역문': tensor([[   2,   46,   16,  ...,    1,    1,    1],
            [   2,   86,    4,  ...,    1,    1,    1],
            [   2,    4,  984,  ...,    1,    1,    1],
            ...,
            [   2, 1400,   10,  ...,    1,    1,    1],
            [   2,    4, 7788,  ...,    1,    1,    1],
            [   2,   12,    4,  ...,    1,    1,    1]], device='cuda:0')}



```python
from torchtext.data import Field, TabularDataset, Iterator

# 필드 정의
text_field = Field(sequential=True, tokenize='basic_english')
label_field = Field(sequential=False, dtype=torch.long)

# 데이터셋 생성
train_data, test_data = TabularDataset.splits(
    path='data/',
    train='train.csv',
    test='test.csv',
    format='csv',
    fields=[('text', text_field), ('label', label_field)]
)

# Iterator 생성
train_iterator, test_iterator = Iterator.splits(
    (train_data, test_data),
    batch_size=32,
    shuffle=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 데이터 로드 예시
for batch in train_iterator:
    inputs, labels = batch.text, batch.label
    # 모델 학습 등의 작업 수행

```
