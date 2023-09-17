from flask import Flask, jsonify, request
import openai
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return 'This is home!'

@app.route('/empathy', methods=['POST'])
def empathy():
    user_content = request.get_json()['userContent']
    if(user_content != ""):
        # Load your API key from an environment variable or secret management service
        openai.api_key = "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" # 다른 사람이 알면 안된다! 나중에 변수로 수정하기

        messages=[{"role": "system", "content": "내가 오늘 어떤 일이 있었는지 그리고 어떤 감정과 생각이 들었는지 말할거야. 그럼 너는 감정적으로 내 이야기에 공감하고 위로해주는 문장을 한두개로 요약해서 출력해줘. 한국어로 출력해줘"}]

        #user_content = "배가 고프다.. 배가 고픈데도 먹을것을 고민하고 먹는게 귀찮아서 먹지 않았다. 점점 생명의 불이 꺼져간다."
        messages.append({"role":"user", "content":f"{user_content}"})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages)

        assistant_content = completion.choices[0].message["content"].strip()
        print("USER:", user_content)
        print("GPT:", assistant_content)
        return jsonify({"empathyBot":assistant_content})

print("osssssssss: ", os.getcwd())


def createContentList(data):
    contentList = []
    for i in range(0, len(data)):
        contentList.append(data[i]['content'])
    return contentList

def nameToIndex(data):
    name_to_index = {}
    for i in range(0, len(data)):
        name_to_index[data[i]['name']] = i
    return name_to_index

def get_recommendations(target_data, data, stop_word_list):

    print("dataaaaaaaaaaaaa", data)
    # 선택한 영화의 타이틀로부터 해당 영화의 인덱스를 받아온다.
    contentList = createContentList(data)
    name_to_index = nameToIndex(data)

    tfidfVector = TfidfVectorizer(stop_words=stop_word_list)
    tfidf_matrix = tfidfVector.fit_transform(contentList)
    #print("단어사전", tfidfVector.vocabulary_)
    print('TF-IDF 행렬의 크기(shape)2 :',tfidf_matrix.shape)

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print('코사인 유사도 연산 결과2 :',cosine_sim) # 상호 유사도 기록되어져 있다!

    idx = name_to_index[target_data['name']]

    # 해당 영화와 모든 영화와의 유사도를 가져온다.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 영화들을 정렬한다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 3개의 명언을 받아온다.
    sim_scores = sim_scores[1:4]
    print("sim_scores")

    # 가장 유사한 10개의 명언의 인덱스를 얻는다.
    indices = [idx[0] for idx in sim_scores]
    # 가장 유사한 10개의 명언의 작가 이름을 리턴한다.
    print('* 명언 추천')
    wise_saying_best3=[]
    for i in range(0, len(indices)):
        print(str(i+1) + "등:"
            , data[indices[i]]['name']+": "+data[indices[i]]['content']
            , ", 카테고리: "+data[indices[i]]['category']
            , ", 유사도: ",sim_scores[i][1])
        wise_saying_ = {"rank":i+1, "name":data[indices[i]]['name'],"content":data[indices[i]]['content']}
        wise_saying_best3.append(wise_saying_)
    return wise_saying_best3
# sorted(리스트): 본체 리스트는 내버려두고, 정렬한 새로운 리스트를 반환

@app.route('/wise-saying', methods=['POST'])
def wise_saying():
    
    print("osssssssss: ", os.getcwd())

    getJson = request.get_json()
    user_content = getJson["userContent"]
    print("user_content: ", user_content)
    target_data = {
        'name': '사용자',
        'content': user_content,
        'category': "None"
    }

    # 1. 데이터 불러오기
    with open('home/ubuntu/quotes_kor_data.json', 'r', encoding='UTF8') as file:
        data_kor = json.load(file)
    data_kor.append(target_data)
    print("한글 명언 수: ", len(data_kor))
    
    # 2. 불용어 불러오기
    with open('home/ubuntu/stop_word.json', 'r', encoding='utf-8') as file:
        stop_word_list_kor = json.load(file)

    print("제발;;;")

    # 3. TF-IDF 구하기, 명언 3개 추천
    wise_saying_best3 = get_recommendations(target_data, data_kor, stop_word_list_kor)

    return jsonify({"wiseSayingBest3":wise_saying_best3})

if __name__ == '__main__':
    app.run(debug=True,host="127.0.0.1",port=5000)