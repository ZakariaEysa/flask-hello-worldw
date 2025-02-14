from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# تحميل البيانات من CSV
merged_data = pd.read_csv('movie.csv')

# دمج الميزات للحصول على مرشح المحتوى
merged_data['combined_features'] = merged_data[['genres', 'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']].fillna('').agg(' '.join, axis=1)

# تنظيف البيانات: إزالة الصفوف التي تحتوي على قيم فارغة في العمود combined_features
merged_data.dropna(subset=['combined_features'], inplace=True)

# إنشاء TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(merged_data['combined_features'])

# وظيفة للحصول على توصيات بناءً على وصف
def get_recommendations_by_description(description, num_recommendations=5):
    # تحويل الوصف المدخل إلى متجه باستخدام TF-IDF
    description_vector = tfidf_vectorizer.transform([description])

    # حساب التشابه بين الوصف المدخل وكل الأفلام
    cosine_sim = cosine_similarity(description_vector, tfidf_matrix)

    # الحصول على أعلى النتائج
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:num_recommendations]

    # الحصول على indices الأفلام الموصى بها
    movie_indices = [i[0] for i in sim_scores]
    return merged_data.iloc[movie_indices][['movie_title', 'genres', 'director_name', 'rating']]

# تعريف API endpoint
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    description = data.get('query')

    if not description:
        return jsonify({"error": "Please provide a query string."}), 400

    recommendations = get_recommendations_by_description(description)
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
