from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import requests

app = Flask(__name__)

# Cargar datos de pelis
movies_df = pd.read_csv("movies.csv")
genres_set = set(genre for sublist in movies_df['genres'].str.split('|') for genre in sublist)

# Crear vectores de géneros para cada película
movies_df['genre_vector'] = movies_df['genres'].apply(
    lambda x: [1 if genre in x.split('|') else 0 for genre in genres_set]
)

@app.route("/recommendations/<user_id>", methods=["GET"])
def recommend_for_user(user_id):
    # Obtener géneros del usuario desde un servicio externo
    response = requests.get("http://worker:8080/mysql/users")
    
    if response.status_code != 200:
        return jsonify({"error": "Error al obtener datos de usuarios"}), 500

    # Buscar al usuario por ID
    users_data = response.json()
    user = next((u for u in users_data if u["user_id"] == user_id), None)

    if not user:
        return jsonify({"error": f"No se encontró usuario con ID: {user_id}"}), 404

    # Convertir géneros del usuario a un vector
    user_genres = user["genres"].split(',')
    user_vector = [1 if genre in user_genres else 0 for genre in genres_set]

    # Calcular similitud del coseno para cada película
    movies_df['similarity'] = movies_df['genre_vector'].apply(
        lambda x: 1 - cosine(user_vector, x)  # 1 - coseno para obtener similitud
    )

    # Obtener las películas con mayor similitud (ordenar y tomar las mejores)
    recommended_movies = movies_df.sort_values(by='similarity', ascending=False).head(10)['title'].tolist()

    return jsonify({
        "user_id": user_id,
        "recommended_movies": recommended_movies
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
