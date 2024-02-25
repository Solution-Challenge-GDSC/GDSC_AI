from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# 저장된 모델 불러오기
model = joblib.load('./best_model.pkl')


# 추천 API 엔드포인트
@app.route('/recommend_play', methods=['POST'])
def recommend_play():
    data = request.json  # 클라이언트에서 보낸 데이터
    child_info = data['child_info']  # 아이 정보
    recommended_play = model.predict([child_info])  # 모델에 데이터 전달
    return jsonify({'recommended_play': recommended_play})

if __name__ == '__main__':
    app.run(debug=True)
