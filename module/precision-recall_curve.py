import global_data as global_data # 전역선언

#from sklearn.feature_extraction.text import CountVectorizer # 문장을 숫자로 변경하기 위함
from sklearn.linear_model import LogisticRegression # 학습을 위한 모델 선언
from sklearn.model_selection import train_test_split # 학습 테스트를 위한 선언
from sklearn.metrics import accuracy_score # 학습 테스트 정확도 확인
import numpy as np # 개수 확인
from sklearn.feature_extraction.text import TfidfVectorizer # TF-IDF 사용
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# 컴퓨터가 학습하기위해 문장을 숫자로 바꾸는 것
# vectorizer = CountVectorizer(ngram_range=(1, 2));
vectorizer = TfidfVectorizer( 
    ngram_range=(1,2),  # 단어 범위를 1개에서 2개로 봄
    min_df=3, # 3개 문서 이상 등장
    max_df=0.9, # 너무 흔한 단어 제거 90%
    sublinear_tf=True
) 
x = vectorizer.fit_transform(global_data.df["text"]); 
y = global_data.df["label"];

# 데이터 분리
x_train, x_test, y_train , y_test, text_trans, text_test = train_test_split(
    x, y, global_data.df["text"], test_size=0.3, random_state=42
)

# 모델 만들고 학습시키기
model = LogisticRegression(max_iter=1000);
model.fit(x_train,y_train);

proba = model.predict_proba(x_test)

# 스팸 확률만 사용
y_scores = proba[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_test, y_scores
)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall (스팸)")
plt.ylabel("Precision (스팸)")
plt.title("Precision-Recall Curve")
plt.show()