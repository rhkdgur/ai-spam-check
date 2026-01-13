import global_data as global_data # 전역선언

from sklearn.feature_extraction.text import CountVectorizer # 문장을 숫자로 변경하기 위함
from sklearn.linear_model import LogisticRegression # 학습을 위한 모델 선언
from sklearn.model_selection import train_test_split # 학습 테스트를 위한 선언
from sklearn.metrics import accuracy_score # 학습 테스트 정확도 확인
import numpy as np # 개수 확인

# 컴퓨터가 학습하기위해 문장을 숫자로 바꾸는 것
vectorizer = CountVectorizer();
x = vectorizer.fit_transform(global_data.df["text"]); 
y = global_data.df["label"];

# 데이터 분리
x_train, x_test, y_train , y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 모델 만들고 학습시키기
model = LogisticRegression();
model.fit(x_train,y_train);

# 테스트 데이터를 통해 예측 정확도 확인
result = model.predict(x_test);
score = accuracy_score(y_test, result);
print("정확도:", score);

# 예측 실패건수 확인
fail_idx = np.where(result != y_test)[0]
print("틀린 개수 : ",len(fail_idx))
print(fail_idx)

# 틀린문자리스트
x_test_text = global_data.df.iloc[y_test.index]["text"].values

for i in fail_idx:
    print("문자:", x_test_text[i])
    print("실제:", "스팸" if y_test.iloc[i] == 1 else "정상")
    print("예측:", "스팸" if result[i] == 1 else "정상")
    print("-" * 50)


# 스팸 판단에 영향준 단어 점수 보기
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]
top_spam_idx = coefficients.argsort()[-10:][::-1]
print("스팸에 가장 영향 큰 단어")

for idx in top_spam_idx:
    print(f"{feature_names[idx]} : {coefficients[idx]:.3f}")

# 정상 점수 보기
top_ham_idx = coefficients.argsort()[:10]

print("\n정상에 가장 영향 큰 단어")
for idx in top_ham_idx:
    print(f"{feature_names[idx]} : {coefficients[idx]:.3f}")