from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# 모델 로드
model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # POST 요청에서 텍스트 가져오기
    data = request.json
    text = data['text']

    # 모델 예측
    predicted_label = model.predict([text])[0]

    # 예측 결과를 JSON 형식으로 반환
    return jsonify({'predicted_label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
