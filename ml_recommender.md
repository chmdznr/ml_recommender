# Worksheet: Algoritma Recommender System dalam Machine Learning

## 1. Pengantar Recommender System

Recommender system adalah algoritma yang bertujuan untuk memprediksi preferensi pengguna terhadap item tertentu berdasarkan data historis atau informasi tentang item dan pengguna. Sistem ini telah menjadi bagian penting dalam berbagai aplikasi seperti e-commerce, layanan streaming musik/video, dan platform konten lainnya.

### 1.1 Mengapa Recommender System Penting?

- **Personalisasi**: Meningkatkan pengalaman pengguna dengan konten yang relevan
- **Peningkatan Engagement**: Pengguna cenderung berinteraksi lebih lama dengan sistem yang memberikan rekomendasi yang relevan
- **Peningkatan Konversi**: Membantu pengguna menemukan produk/layanan yang mereka butuhkan lebih cepat

### 1.2 Jenis Utama Recommender System

1. **Non-Personalized Recommender**: Rekomendasi yang sama untuk semua pengguna
2. **Content-Based Filtering**: Rekomendasi berdasarkan karakteristik item
3. **Collaborative Filtering**: Rekomendasi berdasarkan perilaku pengguna lain yang serupa
4. **Matrix Factorization**: Teknik decomposition matrix untuk rekomendasi
5. **Hybrid Methods**: Kombinasi dari pendekatan di atas

Mari kita bahas secara bertahap dengan implementasi Python.

## 2. Non-Personalized Recommender

Ini adalah pendekatan paling sederhana dalam recommender system, di mana rekomendasi tidak disesuaikan dengan preferensi individu.

### Tugas 1: Eksplorasi Dataset
Sebelum mulai membangun model, penting untuk memahami data yang akan kita gunakan:

1. Download dataset MovieLens Small (ml-latest-small) dari https://grouplens.org/datasets/movielens/
2. Lakukan eksplorasi dan analisis data untuk menjawab pertanyaan berikut:
   - Berapa jumlah user, item (film), dan rating dalam dataset?
   - Bagaimana distribusi rating? (buat histogram)
   - Siapa pengguna yang paling aktif memberi rating?
   - Film apa yang mendapatkan rating terbanyak?
   - Bagaimana distribusi genre film dalam dataset?
3. Visualisasikan hasil analisis Anda dengan matplotlib atau seaborn
4. Buatlah kesimpulan dari hasil eksplorasi Anda

### 2.1 Popularity-Based Recommendation

Sistem ini merekomendasikan item yang paling populer (berdasarkan jumlah interaksi, rating, atau views).

#### Konsep dan Rumus:
- Rating rata-rata: $\bar{r}_i = \frac{\sum_{u} r_{ui}}{n_i}$
- Popularitas: $pop_i = \text{count}(r_{ui})$

dimana:
- $r_{ui}$ adalah rating dari pengguna $u$ untuk item $i$
- $n_i$ adalah jumlah pengguna yang memberikan rating untuk item $i$

#### Implementasi Python:

```python
import pandas as pd
import numpy as np

# Load dataset (gunakan MovieLens sebagai contoh)
# Dataset dapat diunduh dari: https://grouplens.org/datasets/movielens/
# Untuk contoh ini, kita akan menggunakan versi kecil: ml-latest-small

# Baca data ratings
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Hitung popularitas berdasarkan jumlah rating
movie_popularity = ratings.groupby('movieId').size().reset_index(name='count')
movie_popularity = movie_popularity.sort_values('count', ascending=False)

# Gabungkan dengan data film untuk mendapatkan judul
popular_movies = movie_popularity.merge(movies, on='movieId')

# Tampilkan 10 film terpopuler
print("10 Film Terpopuler Berdasarkan Jumlah Rating:")
print(popular_movies[['title', 'count']].head(10))

# Hitung popularitas berdasarkan rating rata-rata (minimal 100 rating)
movie_ratings = ratings.groupby('movieId').agg(
    mean_rating=('rating', 'mean'),
    count=('rating', 'count')
).reset_index()

# Filter film dengan minimal 100 rating
popular_by_rating = movie_ratings[movie_ratings['count'] >= 100].sort_values('mean_rating', ascending=False)
popular_by_rating = popular_by_rating.merge(movies, on='movieId')

print("\n10 Film Terpopuler Berdasarkan Rating Rata-rata (min. 100 rating):")
print(popular_by_rating[['title', 'mean_rating', 'count']].head(10))
```

### 2.2 Trending Items

Rekomendasi berdasarkan item yang sedang tren (popularitas dengan bobot waktu).

#### Konsep dan Rumus:
- Trending score: $trend_i = \sum_{u} w(t_{now} - t_{ui}) \cdot r_{ui}$

dimana:
- $t_{ui}$ adalah waktu interaksi pengguna $u$ dengan item $i$
- $t_{now}$ adalah waktu sekarang
- $w(t)$ adalah fungsi bobot yang memberikan nilai lebih tinggi untuk interaksi yang lebih baru

### Tugas 2: Implementasi Non-Personalized Recommender
Pada tugas ini, Anda akan mengimplementasikan dan memodifikasi algoritma non-personalized recommender:

1. Gunakan dataset [Steam Video Games](https://www.kaggle.com/datasets/tamber/steam-video-games) dari Kaggle yang berisi data penggunaan game di platform Steam
2. Implementasikan dua jenis non-personalized recommender:
   - **Most Popular**: Temukan game paling populer berdasarkan jumlah jam bermain
   - **Trending Games**: Buat algoritma trending yang memberi bobot lebih tinggi untuk game yang dimainkan baru-baru ini
3. Modifikasi fungsi bobot waktu dengan 3 pendekatan berbeda:
   - Linear decay
   - Exponential decay
   - Custom decay function (rancangan Anda sendiri)
4. Bandingkan hasil rekomendasi dari ketiga pendekatan tersebut
5. Buatlah visualisasi yang menunjukkan perbedaan ranking game berdasarkan ketiga pendekatan tersebut

#### Implementasi Python:

```python
import pandas as pd
import numpy as np
from datetime import datetime

# Asumsikan ratings.csv memiliki kolom timestamp yang menyimpan waktu rating
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

# Tentukan tanggal referensi sebagai "sekarang"
now = datetime.now()

# Fungsi untuk menghitung bobot berdasarkan usia (dalam hari)
def time_weight(timestamp):
    age_days = (now - timestamp).days
    # Fungsi bobot eksponensial sederhana
    return np.exp(-0.01 * age_days)

# Hitung trending score
ratings['time_weight'] = ratings['timestamp'].apply(time_weight)
ratings['weighted_rating'] = ratings['rating'] * ratings['time_weight']

# Agregat trending score per film
trending_movies = ratings.groupby('movieId').agg(
    trending_score=('weighted_rating', 'sum'),
    count=('movieId', 'count')
).reset_index()

# Filter film dengan minimal 20 rating
trending_movies = trending_movies[trending_movies['count'] >= 20].sort_values('trending_score', ascending=False)
trending_movies = trending_movies.merge(movies, on='movieId')

print("10 Film Tren Teratas:")
print(trending_movies[['title', 'trending_score', 'count']].head(10))
```

## 3. Content-Based Filtering

Content-based filtering memberikan rekomendasi berdasarkan kesamaan antara item dengan preferensi pengguna terhadap item sebelumnya.

### 3.1 TF-IDF untuk Representasi Item

Term Frequency-Inverse Document Frequency (TF-IDF) adalah metode statistik untuk mengukur pentingnya sebuah kata dalam dokumen.

#### Konsep dan Rumus:
- Term Frequency (TF): $TF(t,d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$
- Inverse Document Frequency (IDF): $IDF(t) = \log \frac{N}{DF(t)}$
- TF-IDF: $TFIDF(t,d) = TF(t,d) \times IDF(t)$

dimana:
- $f_{t,d}$ adalah frekuensi term $t$ dalam dokumen $d$
- $N$ adalah jumlah total dokumen
- $DF(t)$ adalah jumlah dokumen yang mengandung term $t$

#### Implementasi Python:

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Gunakan data genre dan overview film untuk content-based filtering
movies = pd.read_csv('ml-latest-small/movies.csv')

# Ekstrak genre
movies['genres'] = movies['genres'].str.replace('|', ' ')

# Gunakan TF-IDF untuk mengekstrak fitur dari genre
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Konversi matrix TF-IDF ke dalam format DataFrame untuk eksplorasi
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out(), index=movies['title'])

print("Representasi TF-IDF untuk 5 film pertama (fitur genre):")
print(tfidf_df.head())
```

### 3.2 Cosine Similarity untuk Rekomendasi

Cosine similarity mengukur kesamaan antara dua vektor dan menentukan apakah mereka menunjuk ke arah yang sama, terlepas dari besarnya.

#### Konsep dan Rumus:
- Cosine Similarity: $cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{|\mathbf{A}||\mathbf{B}|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$

#### Implementasi Python:

```python
# Hitung cosine similarity antara semua pasangan film
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Konversi ke DataFrame untuk kemudahan pembacaan
cosine_sim_df = pd.DataFrame(cosine_sim, index=movies['title'], columns=movies['title'])

# Fungsi untuk mendapatkan rekomendasi berdasarkan kesamaan konten
def get_content_based_recommendations(title, cosine_sim_df, n=10):
    # Dapatkan skor kesamaan untuk semua film dengan film yang ditentukan
    sim_scores = cosine_sim_df[title]
    
    # Urutkan film berdasarkan skor kesamaan
    sim_scores = sim_scores.sort_values(ascending=False)
    
    # Ambil top n rekomendasi (kecuali film itu sendiri yang akan memiliki skor 1.0)
    recommendations = sim_scores.iloc[1:n+1]
    
    return recommendations

# Contoh: Dapatkan rekomendasi untuk film "Toy Story"
print("\nRekomendasi untuk 'Toy Story' berdasarkan genre:")
print(get_content_based_recommendations("Toy Story (1995)", cosine_sim_df))
```

### 3.3 Content-Based dengan Fitur yang Lebih Kompleks

Pada contoh sebelumnya, kita hanya menggunakan genre. Sekarang kita akan mencoba menggabungkan beberapa fitur.

### Tugas 3: Content-Based Recommender untuk Dataset Buku

Pada tugas ini, Anda akan mengimplementasikan content-based filtering untuk dataset buku:

1. Download [Goodbooks-10k dataset](https://github.com/zygmuntz/goodbooks-10k) yang berisi data 10,000 buku populer
2. Fokus pada file `books.csv` yang berisi metadata buku seperti judul, penulis, dan tag
3. Implementasikan content-based filtering dengan langkah-langkah berikut:
   - Ekstrak fitur penting dari data buku (judul, penulis, tag)
   - Gunakan TF-IDF untuk mengubah data tekstual menjadi representasi vektor
   - Implementasikan 3 varian cosine similarity:
     a. Berdasarkan penulis saja
     b. Berdasarkan tag saja
     c. Kombinasi semua fitur dengan pembobotan yang Anda tentukan
4. Buat fungsi yang menerima judul buku dan mengembalikan 10 rekomendasi buku serupa
5. Analisis dan bandingkan hasil rekomendasi dari ketiga pendekatan tersebut
6. Bonus: Tambahkan fitur untuk menggabungkan preferensi pengguna (misalnya, rekomendasi berdasarkan beberapa buku favorit)

```python
# Asumsikan kita memiliki data tambahan seperti deskripsi film, sutradara, dll.
# Kita akan membuat data dummy untuk contoh ini

# Tambahkan kolom overview dummy
np.random.seed(42)
movies['overview'] = ["This movie is about " + " ".join(genres.split("|")) 
                      for genres in movies['genres'].str.replace('|', ' ')]

# Buat fitur kombinasi
movies['content'] = movies['title'] + ' ' + movies['genres'] + ' ' + movies['overview']

# Gunakan TF-IDF untuk mengekstrak fitur
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])

# Hitung cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=movies['title'], columns=movies['title'])

print("\nRekomendasi untuk 'Toy Story' berdasarkan konten yang lebih lengkap:")
print(get_content_based_recommendations("Toy Story (1995)", cosine_sim_df))
```

## 4. Collaborative Filtering

Collaborative filtering memberikan rekomendasi berdasarkan kesamaan perilaku atau preferensi antar pengguna atau antar item.

### 4.1 User-Based Collaborative Filtering

Rekomendasi berdasarkan kesamaan antar pengguna.

#### Konsep dan Rumus:
- Similarity antara pengguna $u$ dan $v$: $sim(u,v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}}$
- Prediksi rating: $\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N_i(u)} sim(u,v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in N_i(u)} |sim(u,v)|}$

dimana:
- $I_{uv}$ adalah himpunan item yang dirating oleh pengguna $u$ dan $v$
- $\bar{r}_u$ adalah rating rata-rata dari pengguna $u$
- $N_i(u)$ adalah himpunan neighbor (pengguna serupa) dari pengguna $u$ yang telah merating item $i$

#### Implementasi Python:

```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Buat user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Fungsi untuk menghitung kesamaan antar pengguna
def compute_user_similarity(user_item_matrix):
    # Menggunakan cosine similarity
    similarity = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(similarity, 
                               index=user_item_matrix.index, 
                               columns=user_item_matrix.index)
    return similarity_df

# Hitung user similarity
user_similarity = compute_user_similarity(user_item_matrix)

# Fungsi untuk memberikan rekomendasi berdasarkan user-based collaborative filtering
def user_based_recommendations(user_id, user_item_matrix, user_similarity, n_users=10, n_items=10):
    # Dapatkan n pengguna paling mirip
    similar_users = user_similarity[user_id].sort_values(ascending=False)[1:n_users+1]
    
    # Mendapatkan item yang belum dirating oleh pengguna
    items_to_recommend = user_item_matrix.columns[user_item_matrix.loc[user_id] == 0]
    
    # Membuat rekomendasi
    recommendations = {}
    
    for item in items_to_recommend:
        # Hitung prediksi rating untuk item
        item_ratings = user_item_matrix[item]
        weighted_ratings = 0
        similarity_sum = 0
        
        for similar_user, similarity in similar_users.items():
            if item_ratings[similar_user] > 0:  # Jika user memberikan rating
                weighted_ratings += similarity * item_ratings[similar_user]
                similarity_sum += similarity
        
        if similarity_sum > 0:
            predicted_rating = weighted_ratings / similarity_sum
            recommendations[item] = predicted_rating
    
    # Urutkan berdasarkan prediksi rating
    recommendations = pd.Series(recommendations).sort_values(ascending=False).head(n_items)
    
    # Dapatkan detail film
    recommended_movies = movies[movies['movieId'].isin(recommendations.index)]
    recommended_movies = recommended_movies.set_index('movieId')
    recommended_movies['predicted_rating'] = recommendations
    
    return recommended_movies[['title', 'predicted_rating']]

# Contoh: Rekomendasi untuk pengguna dengan ID 1
print("\nRekomendasi untuk Pengguna 1 (User-based Collaborative Filtering):")
print(user_based_recommendations(1, user_item_matrix, user_similarity))
```

### 4.2 Item-Based Collaborative Filtering

Rekomendasi berdasarkan kesamaan antar item.

#### Konsep dan Rumus:
- Similarity antara item $i$ dan $j$: $sim(i,j) = \frac{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_i)(r_{uj} - \bar{r}_j)}{\sqrt{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_i)^2} \sqrt{\sum_{u \in U_{ij}} (r_{uj} - \bar{r}_j)^2}}$
- Prediksi rating: $\hat{r}_{ui} = \frac{\sum_{j \in N_u(i)} sim(i,j) \cdot r_{uj}}{\sum_{j \in N_u(i)} |sim(i,j)|}$

dimana:
- $U_{ij}$ adalah himpunan pengguna yang merating item $i$ dan $j$
- $\bar{r}_i$ adalah rating rata-rata untuk item $i$
- $N_u(i)$ adalah himpunan neighbor (item serupa) dari item $i$ yang telah dirating oleh pengguna $u$

### Tugas 4: Collaborative Filtering untuk Rekomendasi Musik

Pada tugas ini, Anda akan mengimplementasikan collaborative filtering untuk dataset musik:

1. Download [Last.fm Dataset (1K users)](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html) yang berisi data pendengar musik
2. Fokus pada file `user_artists.dat` yang berisi hubungan antara pengguna dan artis
3. Implementasikan kedua jenis collaborative filtering:
   - **User-based**: Rekomendasi berdasarkan pengguna serupa
   - **Item-based**: Rekomendasi berdasarkan artis serupa
4. Eksperimen dengan parameter algoritma:
   - Jumlah neighbor (k): 5, 10, 20, 50
   - Metode similarity: Cosine, Pearson correlation, Jaccard
   - Threshold minimal interaksi: 2, 5, 10
5. Untuk setiap konfigurasi:
   - Evaluasi performa dengan metode cross-validation
   - Hitung metrik precision@k dan recall@k
6. Identifikasi konfigurasi terbaik dan analisis hasilnya
7. Bonus: Implementasikan metode untuk mengatasi cold-start problem (pengguna baru)

#### Implementasi Python:

```python
# Transpose user-item matrix untuk mendapatkan item-user matrix
item_user_matrix = user_item_matrix.T

# Hitung item similarity
def compute_item_similarity(item_user_matrix):
    # Menggunakan cosine similarity
    similarity = cosine_similarity(item_user_matrix)
    similarity_df = pd.DataFrame(similarity, 
                               index=item_user_matrix.index, 
                               columns=item_user_matrix.index)
    return similarity_df

item_similarity = compute_item_similarity(item_user_matrix)

# Fungsi untuk memberikan rekomendasi berdasarkan item-based collaborative filtering
def item_based_recommendations(user_id, user_item_matrix, item_similarity, n_items=10, n_similar=10):
    # Dapatkan item yang sudah dirating oleh pengguna
    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index
    
    # Dapatkan item yang belum dirating
    items_to_recommend = user_item_matrix.columns[user_item_matrix.loc[user_id] == 0]
    
    # Hitung prediksi rating untuk item yang belum dirating
    recommendations = {}
    
    for item_to_rec in items_to_recommend:
        # Dapatkan item serupa yang telah dirating pengguna
        similar_items = item_similarity[item_to_rec].loc[rated_items].sort_values(ascending=False).head(n_similar)
        
        # Hitung prediksi rating
        weighted_ratings = 0
        similarity_sum = 0
        
        for similar_item, similarity in similar_items.items():
            if similarity > 0:  # Positif similarity
                rating = user_ratings[similar_item]
                weighted_ratings += similarity * rating
                similarity_sum += similarity
        
        if similarity_sum > 0:
            predicted_rating = weighted_ratings / similarity_sum
            recommendations[item_to_rec] = predicted_rating
    
    # Urutkan berdasarkan prediksi rating
    recommendations = pd.Series(recommendations).sort_values(ascending=False).head(n_items)
    
    # Dapatkan detail film
    recommended_movies = movies[movies['movieId'].isin(recommendations.index)]
    recommended_movies = recommended_movies.set_index('movieId')
    recommended_movies['predicted_rating'] = recommendations
    
    return recommended_movies[['title', 'predicted_rating']]

# Contoh: Rekomendasi untuk pengguna dengan ID 1
print("\nRekomendasi untuk Pengguna 1 (Item-based Collaborative Filtering):")
print(item_based_recommendations(1, user_item_matrix, item_similarity))
```

## 5. Matrix Factorization

Matrix factorization adalah teknik untuk mendekomposisi matriks user-item menjadi dua matriks berukuran lebih kecil yang merepresentasikan user dan item dalam ruang "laten".

### 5.1 Singular Value Decomposition (SVD)

SVD adalah salah satu teknik matrix factorization yang populer.

#### Konsep dan Rumus:
- SVD: $R \approx U \Sigma V^T$

dimana:
- $R$ adalah matriks rating user-item
- $U$ adalah matriks pengguna-faktor
- $\Sigma$ adalah matriks diagonal dengan singular value
- $V^T$ adalah matriks item-faktor transpos

#### Implementasi Python:

```python
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error

# Load dataset
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Buat user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Fungsi untuk menerapkan SVD
def apply_svd(user_item_matrix, k=20):
    # Standardize matriks
    ratings_mean = np.mean(user_item_matrix.values, axis=1)
    ratings_demeaned = user_item_matrix.values - ratings_mean.reshape(-1, 1)
    
    # Terapkan SVD
    U, sigma, Vt = svds(ratings_demeaned, k=k)
    
    # Urutkan singular values
    sigma_diag = np.diag(sigma)
    
    # Rekonstruksi matriks rating
    predicted_ratings = U.dot(sigma_diag).dot(Vt) + ratings_mean.reshape(-1, 1)
    predicted_df = pd.DataFrame(predicted_ratings, 
                              index=user_item_matrix.index, 
                              columns=user_item_matrix.columns)
    
    return predicted_df, U, sigma, Vt

# Terapkan SVD dengan 20 faktor laten
predicted_ratings, U, sigma, Vt = apply_svd(user_item_matrix, k=20)

# Fungsi untuk memberikan rekomendasi berdasarkan matriks prediksi
def svd_recommendations(user_id, predicted_ratings, user_item_matrix, n_items=10):
    # Dapatkan prediksi rating untuk pengguna
    user_predictions = predicted_ratings.loc[user_id]
    
    # Dapatkan item yang belum dirating
    items_to_recommend = user_item_matrix.columns[user_item_matrix.loc[user_id] == 0]
    
    # Filter prediksi untuk item yang belum dirating
    predicted_ratings_unrated = user_predictions[items_to_recommend]
    
    # Urutkan berdasarkan prediksi rating
    recommendations = predicted_ratings_unrated.sort_values(ascending=False).head(n_items)
    
    # Dapatkan detail film
    recommended_movies = movies[movies['movieId'].isin(recommendations.index)]
    recommended_movies = recommended_movies.set_index('movieId')
    recommended_movies['predicted_rating'] = recommendations
    
    return recommended_movies[['title', 'predicted_rating']]

# Contoh: Rekomendasi untuk pengguna dengan ID 1
print("\nRekomendasi untuk Pengguna 1 (Matrix Factorization dengan SVD):")
print(svd_recommendations(1, predicted_ratings, user_item_matrix))

# Evaluasi performa SVD dengan RMSE
# Pisahkan data menjadi train dan test
from sklearn.model_selection import train_test_split

ratings_train, ratings_test = train_test_split(ratings, test_size=0.2, random_state=42)

# Buat matriks rating untuk training
user_item_train = ratings_train.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Terapkan SVD ke data training
predicted_ratings_train, _, _, _ = apply_svd(user_item_train, k=20)

# Evaluasi pada data test
test_user_item_pairs = list(zip(ratings_test['userId'], ratings_test['movieId']))
predicted = []
actual = []

for user_id, movie_id in test_user_item_pairs:
    if user_id in predicted_ratings_train.index and movie_id in predicted_ratings_train.columns:
        predicted.append(predicted_ratings_train.loc[user_id, movie_id])
        actual.append(ratings_test[(ratings_test['userId'] == user_id) & 
                                 (ratings_test['movieId'] == movie_id)]['rating'].values[0])

# Hitung RMSE
rmse = np.sqrt(mean_squared_error(actual, predicted))
print(f"\nRoot Mean Squared Error (RMSE) for SVD: {rmse:.4f}")
```

### 5.2 Alternating Least Squares (ALS)

ALS adalah algoritma matrix factorization yang memecahkan masalah optimasi dengan cara bergantian memperbaiki faktor pengguna dan faktor item.

#### Konsep dan Rumus:
ALS meminimalkan fungsi biaya:
$\min_{P,Q} \sum_{(u,i) \in K} (r_{ui} - p_u^T q_i)^2 + \lambda (||p_u||^2 + ||q_i||^2)$

dimana:
- $p_u$ adalah vektor faktor laten untuk pengguna $u$
- $q_i$ adalah vektor faktor laten untuk item $i$
- $\lambda$ adalah parameter regularisasi
- $K$ adalah himpunan pasangan (pengguna, item) dengan rating yang diketahui

### Tugas 5: Matrix Factorization untuk E-commerce

Pada tugas ini, Anda akan mengimplementasikan matrix factorization untuk data e-commerce:

1. Download [Retail Rocket Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) yang berisi data interaksi pengguna dengan produk e-commerce
2. Lakukan preprocessing data:
   - Konversi event type (view, add-to-cart, transaction) menjadi nilai numerik
   - Buat user-item matrix berdasarkan interaksi
3. Implementasikan dua algoritma matrix factorization:
   - **SVD**: Menggunakan pendekatan SVD dari scipy
   - **ALS**: Menggunakan library implicit atau implementasi manual
4. Eksperimen dengan hyperparameter:
   - Jumlah faktor laten: 10, 20, 50, 100
   - Regularisasi (λ): 0.01, 0.1, 1.0
   - Iterasi: 10, 20, 50
5. Evaluasi dan bandingkan performa kedua algoritma menggunakan:
   - Metrik RMSE
   - Metrik ranking (precision@k, recall@k)
   - Waktu komputasi
6. Visualisasikan latent factors untuk mengetahui pola tersembunyi dalam data:
   - Gunakan t-SNE atau PCA untuk visualisasi
   - Identifikasi cluster yang muncul dari latent factors
7. Bonus: Implementasikan BPR (Bayesian Personalized Ranking) untuk optimasi ranking

#### Implementasi Python dengan Implicit:

```python
import pandas as pd
import numpy as np
import implicit
from scipy.sparse import csr_matrix

# Load dataset
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Buat sparse matrix untuk user-item interactions
user_ids = ratings['userId'].astype("category").cat.codes
movie_ids = ratings['movieId'].astype("category").cat.codes
movie_id_mapping = dict(zip(movie_ids, ratings['movieId']))
user_id_mapping = dict(zip(user_ids, ratings['userId']))

# Buat sparse matrix
sparse_user_item = csr_matrix((ratings['rating'].values, (user_ids, movie_ids)))

# Fit ALS model
model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
model.fit(sparse_user_item.T)  # Transpose untuk item-user matrix

# Fungsi untuk mendapatkan rekomendasi
def als_recommendations(user_id, model, sparse_user_item, user_id_mapping, movie_id_mapping, 
                        movies_df, n_items=10):
    # Dapatkan user_id internal
    user_idx = list(user_id_mapping.keys())[list(user_id_mapping.values()).index(user_id)]
    
    # Dapatkan rekomendasi
    recommendations = model.recommend(user_idx, sparse_user_item[user_idx], N=n_items)
    
    # Convert internal item indices ke movie IDs
    movie_indices = [idx for idx, _ in recommendations]
    movie_ids = [movie_id_mapping[idx] for idx in movie_indices]
    scores = [score for _, score in recommendations]
    
    # Dapatkan detail film
    recommended_movies = movies_df[movies_df['movieId'].isin(movie_ids)]
    recommended_movies['score'] = scores
    
    return recommended_movies[['title', 'genres', 'score']]

# Contoh: Rekomendasi untuk pengguna dengan ID 1
print("\nRekomendasi untuk Pengguna 1 (Matrix Factorization dengan ALS):")
print(als_recommendations(1, model, sparse_user_item, user_id_mapping, movie_id_mapping, movies))
```

## 6. Hybrid Methods

Hybrid recommender systems menggabungkan berbagai jenis algoritma rekomendasi untuk meningkatkan performa dan mengatasi kelemahan masing-masing pendekatan.

### 6.1 Weighted Hybrid

Menggabungkan hasil dari beberapa sistem rekomendasi dengan memberikan bobot.

#### Implementasi Python:

```python
# Fungsi untuk membuat hybrid recommender dengan weighted average
def weighted_hybrid_recommendations(user_id, content_weight=0.3, collab_weight=0.7, n_items=10):
    # Dapatkan rekomendasi content-based
    # Asumsikan kita memiliki fungsi untuk mendapatkan film yang disukai user
    favorite_movie = "Toy Story (1995)"  # Misalnya, ini adalah film favorit user
    content_rec = get_content_based_recommendations(favorite_movie, cosine_sim_df, n=20)
    content_rec = pd.DataFrame(content_rec).reset_index()
    content_rec.columns = ['title', 'content_score']
    
    # Dapatkan rekomendasi collaborative filtering
    collab_rec = item_based_recommendations(user_id, user_item_matrix, item_similarity, n_items=20)
    collab_rec = collab_rec.reset_index()[['title', 'predicted_rating']]
    collab_rec.columns = ['title', 'collab_score']
    
    # Gabungkan rekomendasi
    hybrid_rec = pd.merge(content_rec, collab_rec, on='title', how='outer').fillna(0)
    
    # Normalisasi skor
    if len(hybrid_rec) > 0:
        hybrid_rec['content_score'] = hybrid_rec['content_score'] / hybrid_rec['content_score'].max()
        if hybrid_rec['collab_score'].max() > 0:
            hybrid_rec['collab_score'] = hybrid_rec['collab_score'] / hybrid_rec['collab_score'].max()
    
    # Hitung weighted score
    hybrid_rec['hybrid_score'] = (
        content_weight * hybrid_rec['content_score'] + 
        collab_weight * hybrid_rec['collab_score']
    )
    
    # Urutkan dan ambil top-n
    hybrid_rec = hybrid_rec.sort_values('hybrid_score', ascending=False).head(n_items)
    
    return hybrid_rec[['title', 'content_score', 'collab_score', 'hybrid_score']]

# Contoh: Rekomendasi hybrid untuk pengguna dengan ID 1
print("\nRekomendasi Hybrid untuk Pengguna 1:")
print(weighted_hybrid_recommendations(1))
```

### 6.2 Switching Hybrid

Memilih salah satu sistem rekomendasi berdasarkan kondisi tertentu.

#### Implementasi Python:

```python
# Fungsi untuk membuat switching hybrid recommender
def switching_hybrid_recommendations(user_id, min_ratings_threshold=5, n_items=10):
    # Hitung jumlah rating yang diberikan oleh user
    user_ratings = user_item_matrix.loc[user_id]
    user_rating_count = (user_ratings > 0).sum()
    
    # Jika user memiliki cukup rating, gunakan collaborative filtering
    if user_rating_count >= min_ratings_threshold:
        print(f"User {user_id} memiliki {user_rating_count} ratings. Menggunakan collaborative filtering.")
        return item_based_recommendations(user_id, user_item_matrix, item_similarity, n_items=n_items)
    else:
        # Jika tidak, gunakan content-based
        print(f"User {user_id} hanya memiliki {user_rating_count} ratings. Menggunakan content-based filtering.")
        favorite_movie = "Toy Story (1995)"  # Misalnya, ini adalah film favorit user atau film dengan rating tertinggi
        content_rec = get_content_based_recommendations(favorite_movie, cosine_sim_df, n=n_items)
        return pd.DataFrame(content_rec).reset_index()

# Contoh: Rekomendasi switching hybrid untuk pengguna dengan ID 1
print("\nRekomendasi Switching Hybrid untuk Pengguna 1:")
print(switching_hybrid_recommendations(1))
```

### Tugas 6: Hybrid Recommender untuk Platform Streaming

Pada tugas ini, Anda akan mengembangkan sistem rekomendasi hybrid untuk dataset platform streaming:

1. Download [Netflix Prize Dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) atau [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/)
2. Implementasikan tiga jenis sistem rekomendasi:
   - **Content-based**: Berdasarkan genre, sutradara, aktor, dll.
   - **Collaborative filtering**: Berdasarkan rating pengguna
   - **Matrix factorization**: Menggunakan SVD
3. Kembangkan dua pendekatan hybrid:
   - **Weighted hybrid**: Kombinasi rekomendasi dengan bobot
   - **Feature augmentation**: Gunakan output dari satu sistem sebagai input untuk sistem lain
4. Rancang dan implementasikan strategi switching yang cerdas:
   - Pertimbangkan konteks seperti waktu, jumlah interaksi pengguna, dll.
   - Buat aturan switching yang adaptif
5. Evaluasi semua pendekatan dan bandingkan hasilnya
6. Identifikasi skenario di mana masing-masing pendekatan hybrid memberikan hasil terbaik
7. Bonus: Implementasikan contextual recommendations yang mempertimbangkan waktu dalam seminggu/hari

## 7. Evaluasi Recommender System

Evaluasi adalah bagian penting untuk mengetahui performa sistem rekomendasi.

### 7.1 Metrics untuk Evaluasi

#### Implementasi Python:

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Fungsi untuk mengevaluasi rekomendasi
def evaluate_recommender(predicted_ratings, actual_ratings):
    """
    Evaluasi recommender system dengan berbagai metrik
    
    Parameters:
    predicted_ratings (array): Prediksi rating
    actual_ratings (array): Rating aktual
    
    Returns:
    dict: Dictionary dengan berbagai metrik evaluasi
    """
    # Hitung Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    
    # Hitung Mean Absolute Error (MAE)
    mae = mean_absolute_error(actual_ratings, predicted_ratings)
    
    return {
        "RMSE": rmse,
        "MAE": mae
    }

# Contoh: Evaluasi SVD
# Asumsikan kita sudah membagi data menjadi train dan test
from sklearn.model_selection import train_test_split

ratings_train, ratings_test = train_test_split(ratings, test_size=0.2, random_state=42)

# Buat matriks rating untuk training
user_item_train = ratings_train.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Terapkan SVD ke data training
predicted_ratings_train, _, _, _ = apply_svd(user_item_train, k=20)

# Evaluasi pada data test
test_user_item_pairs = list(zip(ratings_test['userId'], ratings_test['movieId']))
predicted = []
actual = []

for user_id, movie_id in test_user_item_pairs:
    if user_id in predicted_ratings_train.index and movie_id in predicted_ratings_train.columns:
        predicted.append(predicted_ratings_train.loc[user_id, movie_id])
        actual.append(ratings_test[(ratings_test['userId'] == user_id) & 
                                 (ratings_test['movieId'] == movie_id)]['rating'].values[0])

# Evaluasi
eval_results = evaluate_recommender(predicted, actual)
print("\nHasil Evaluasi SVD:")
for metric, value in eval_results.items():
    print(f"{metric}: {value:.4f}")
```

### 7.2 Evaluasi Berbasis Ranking

```python
from sklearn.metrics import ndcg_score, precision_score, recall_score
import numpy as np

# Fungsi untuk mengevaluasi berdasarkan ranking
def evaluate_ranking(recommendations, ground_truth, k=10):
    """
    Evaluasi recommender system berdasarkan ranking
    
    Parameters:
    recommendations (list): Daftar item yang direkomendasikan
    ground_truth (list): Daftar item yang relevan
    k (int): Jumlah top-k item yang dievaluasi
    
    Returns:
    dict: Dictionary dengan metrik evaluasi ranking
    """
    # Precision@k
    if len(recommendations) > 0:
        precision_k = len(set(recommendations[:k]) & set(ground_truth)) / len(recommendations[:k])
    else:
        precision_k = 0
    
    # Recall@k
    if len(ground_truth) > 0:
        recall_k = len(set(recommendations[:k]) & set(ground_truth)) / len(ground_truth)
    else:
        recall_k = 0
    
    # F1 Score
    if precision_k + recall_k > 0:
        f1_score = 2 * (precision_k * recall_k) / (precision_k + recall_k)
    else:
        f1_score = 0
    
    return {
        f"Precision@{k}": precision_k,
        f"Recall@{k}": recall_k,
        f"F1 Score@{k}": f1_score
    }

# Contoh: Evaluasi Ranking untuk user-based collaborative filtering
# Memisahkan data untuk evaluasi
users = ratings['userId'].unique()
results = {}

for user_id in users[:5]:  # Evaluasi untuk 5 pengguna pertama
    # Ambil data rating pengguna
    user_ratings = ratings[ratings['userId'] == user_id]
    
    # Bagi menjadi train (80%) dan test (20%)
    train, test = train_test_split(user_ratings, test_size=0.2, random_state=42)
    
    # Item yang dirating tinggi (>=4) dalam test set adalah ground truth
    ground_truth = test[test['rating'] >= 4]['movieId'].tolist()
    
    # Gunakan data training untuk membuat rekomendasi
    # Persiapkan user-item matrix baru tanpa data test
    temp_ratings = ratings.copy()
    temp_ratings = temp_ratings.drop(test.index)
    temp_user_item = temp_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    # Hitung user similarity
    temp_user_similarity = compute_user_similarity(temp_user_item)
    
    # Dapatkan rekomendasi
    recommendations = user_based_recommendations(user_id, temp_user_item, temp_user_similarity, n_items=20)
    recommended_items = recommendations.index.tolist()
    
    # Evaluasi
    eval_result = evaluate_ranking(recommended_items, ground_truth, k=10)
    results[user_id] = eval_result

# Tampilkan hasil rata-rata
avg_results = {}
for metric in list(results.values())[0].keys():
    avg_results[metric] = np.mean([result[metric] for result in results.values()])

print("\nHasil Evaluasi Ranking (Rata-rata untuk 5 pengguna):")
for metric, value in avg_results.items():
    print(f"{metric}: {value:.4f}")
```

### Tugas 7: Evaluasi Komprehensif dan Deployment

Pada tugas akhir ini, Anda akan melakukan evaluasi menyeluruh dan mempersiapkan model untuk deployment:

1. Pilih dataset yang Anda gunakan pada salah satu tugas sebelumnya
2. Implementasikan berbagai algoritma yang telah dipelajari:
   - Non-personalized recommender
   - Content-based filtering
   - Collaborative filtering (user-based dan item-based)
   - Matrix factorization (SVD dan ALS)
   - Hybrid recommender
3. Rancang metodologi evaluasi yang komprehensif:
   - Split data: training, validation, dan testing
   - Lakukan cross-validation
   - Implementasikan berbagai metrik evaluasi:
     - RMSE, MAE untuk prediksi rating
     - Precision, Recall, F1-score untuk rekomendasi
     - NDCG untuk evaluasi ranking
     - Diversity dan Coverage untuk mengukur variasi rekomendasi
4. Lakukan A/B testing simulasi:
   - Pisahkan pengguna menjadi grup kontrol dan eksperimen
   - Simulasikan rekomendasi untuk setiap grup
   - Analisis perbedaan performa
5. Hasil dan Kesimpulan:
   - Buat visualisasi perbandingan performa semua algoritma
   - Identifikasi algoritma terbaik untuk kasus penggunaan berbeda
   - Berikan rekomendasi untuk implementasi sistem rekomendasi di dunia nyata
6. Bonus: Implementasikan model terbaik dalam bentuk API sederhana menggunakan Flask atau FastAPI

---

## Kesimpulan

Dalam worksheet ini, kita telah mempelajari dan mengimplementasikan berbagai algoritma recommender system mulai dari yang sederhana hingga yang lebih kompleks:

1. **Non-Personalized Recommender**: Popularity-based dan trending items
2. **Content-Based Filtering**: Menggunakan TF-IDF dan cosine similarity
3. **Collaborative Filtering**: User-based dan item-based approaches
4. **Matrix Factorization**: SVD dan ALS untuk mencari faktor laten
5. **Hybrid Methods**: Menggabungkan pendekatan berbeda untuk hasil yang lebih baik
6. **Evaluasi**: Berbagai metrik untuk mengukur performa sistem rekomendasi

Masing-masing algoritma memiliki kelebihan dan kekurangan, dan pemilihan algoritma tergantung pada konteks spesifik dan jenis data yang tersedia.