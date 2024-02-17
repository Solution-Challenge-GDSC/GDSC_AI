from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

# 간단한 텍스트 데이터셋 생성
train_texts = ["안녕하세요", "반갑습니다", "만나서 반가워요", "잘 지내시나요?", "오늘 날씨가 좋네요"]
train_labels = ["인사", "인사", "인사", "인사", "날씨"]

# 텍스트를 벡터화하는 CountVectorizer 및 RandomForestClassifier를 사용하여 파이프라인 구축
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', RandomForestClassifier())
])

# 모델 훈련
pipeline.fit(train_texts, train_labels)

# 테스트용 텍스트 입력
test_texts = ["안녕하세요! 오늘은 비가 올 것 같네요.", "잘 지내고 있어요?"]

# 예측
predicted_labels = pipeline.predict(test_texts)
print("입력 텍스트:", test_texts)
print("예측된 레이블:", predicted_labels)

# 모델 저장
joblib.dump(pipeline, 'model.joblib')
