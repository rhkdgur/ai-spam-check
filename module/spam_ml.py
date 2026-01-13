import global_data as global_data
from sklearn.feature_extraction.text import CountVectorizer # 문장을 숫자로 변경하기 위함
from sklearn.linear_model import LogisticRegression # 학습을 위한 모델 선언

# 컴퓨터가 학습하기위해 문장을 숫자로 바꾸는 것
vectorizer = CountVectorizer();
x = vectorizer.fit_transform(global_data.df["text"]); 
y = global_data.df["label"];

# 모델 만들고 학습시키기
model = LogisticRegression();
model.fit(x,y);

# 테스트 데이터
test_text = ["주식방이 열렸습니다. 참가하세요"];
print(test_text);
test_vecter = vectorizer.transform(test_text);

result = model.predict(test_vecter);
print(result[0]);
print("스팸입니다" if result[0] == 1 else "정상입니다.")