from flask import Flask, render_template, request
import joblib
import pandas as pd

from data_preprocessor import data_preprocessor

app = Flask(__name__)

model_sentiment_analysis = joblib.load('./models/sentiment_analysis_model.pkl')
model_product_recommender = joblib.load('./models/product_recommendation_model.pkl')
tfidf_converter = joblib.load('./models/tfidf_converter.pkl')
df_product_name_reviews = pd.read_csv('./data/df_product_name_review.csv')
preprocessor = data_preprocessor()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend_top_5_products', methods=['POST'])
def recommend_top_5_products():
    # username = request.get_json(force=True).get('username')
    username = request.form.get('username')
    # top 20 products from product-recommendation model
    if username not in model_product_recommender.index.tolist():
        # return {"message": "user - {} not found in our database".format(username)}
        return render_template('index.html',
                               error='User - {} not found in our database'.format(username))
    result = model_product_recommender.loc[username].sort_values(ascending=False)[0:20]
    top_20_products = result.index.values.tolist()
    # return {"top_20_recommended_products": top_20_products}

    # Get sentiment of all the reviews for top 20 products
    dict_product_positive_sentiment_percentage = {}
    for product in top_20_products:
        reviews = df_product_name_reviews[df_product_name_reviews.name == product]['reviews_text'].tolist()
        reviews_preprocessed = []
        for review in reviews:
            reviews_preprocessed.append(' '.join(preprocessor.preprocess_data(sent=review)))
        reviews_tfidf_matrix = tfidf_converter.transform(reviews_preprocessed).toarray()
        predictions = model_sentiment_analysis.predict(reviews_tfidf_matrix)
        positive_sentiments_count = 0
        for sentiment in predictions:
            if sentiment.lower() == 'positive':
                positive_sentiments_count += 1
        percentage_positive_sentiments = (positive_sentiments_count / len(reviews)) * 100
        dict_product_positive_sentiment_percentage[product] = percentage_positive_sentiments

    # Get Top 5 products from dict_product_positive_sentiment_percentage based on positive sentiment %

    sorted_top_20_positive_ratings_products = dict(sorted(dict_product_positive_sentiment_percentage.items(),
                                                          key=lambda x: x[1], reverse=True))
    top_5_recommended_products = list(sorted_top_20_positive_ratings_products.keys())[0:5]
    # return {
    #     "top_5_recommended_products": top_5_recommended_products
    # }

    return render_template('index.html', recommended_products=top_5_recommended_products)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
