import pandas as pd # 엑셀처럼 데이터를 다루기 위함
from sklearn.feature_extraction.text import CountVectorizer # 문장을 숫자로 변경하기 위함
from sklearn.linear_model import LogisticRegression # 학습을 위한 모델 선언

data = {
    "text": [
        "무료 쿠폰 지금 바로 받으세요",
        "오늘 회의는 오후 3시에 시작합니다",
        "지금 가입하면 100% 혜택 지급",
        "엄마 오늘 늦게 들어갈게",
        "이 링크를 클릭하면 보너스 제공"
    ],
    "label": [1, 0, 1, 0, 1] # 1. 스펨 , 0 : 정상
}

df = pd.DataFrame(data)
print(df);

vectorizer = CountVectorizer();
x = vectorizer.fit_transform(df["text"]);
y = df["label"];