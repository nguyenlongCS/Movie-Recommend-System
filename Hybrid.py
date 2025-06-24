import streamlit as st
import pandas as pd
import csv
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from PIL import Image
from io import BytesIO
import requests

# Cấu hình chung
st.set_page_config(page_title="Hệ thống gợi ý phim", layout="wide", initial_sidebar_state="expanded")

# Đường dẫn file lưu thông tin người dùng
USER_DB_FILE = "user.csv"

# Hàm kiểm tra tên đăng nhập trong file CSV
def user_exists(username):
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == username:
                    return True
    return False

# Hàm lưu thông tin người dùng vào file CSV
def save_user(username, password):
    with open(USER_DB_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([username, password])

# Hàm kiểm tra mật khẩu trong file CSV
def check_login(username, password):
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == username and row[1] == password:
                    return True
    return False

# Thanh đăng nhập/đăng ký
st.sidebar.title("Đăng Nhập")
tab = st.sidebar.radio("", ["Đăng Nhập", "Đăng Ký"])

# Kiểm tra nếu người dùng đã đăng nhập
if 'username' not in st.session_state:
    st.session_state.username = None

# Phần đăng nhập
if tab == "Đăng Nhập":
    username = st.sidebar.text_input("Tên đăng nhập")
    password = st.sidebar.text_input("Mật khẩu", type="password")

    if st.sidebar.button("Đăng nhập"):
        if user_exists(username):
            if check_login(username, password):  # So sánh trực tiếp mật khẩu
                st.session_state.username = username
                st.sidebar.success("Đăng nhập thành công!")
            else:
                st.sidebar.error("Mật khẩu sai!")
        else:
            st.sidebar.error("Tên đăng nhập không tồn tại!")

# Phần đăng ký
if tab == "Đăng Ký":
    username = st.sidebar.text_input("Tên đăng nhập")
    password = st.sidebar.text_input("Mật khẩu", type="password")
    confirm_password = st.sidebar.text_input("Nhập lại mật khẩu", type="password")

    if st.sidebar.button("Đăng ký"):
        if password == confirm_password:
            if not user_exists(username):
                save_user(username, password)  # Lưu trực tiếp mật khẩu
                st.sidebar.success("Đăng ký thành công!")
            else:
                st.sidebar.error("Tên đăng nhập đã tồn tại!")
        else:
            st.sidebar.error("Mật khẩu nhập lại không khớp!")

st.markdown("""
    <h1 style='text-align: center; color: white;'>Hệ thống gợi ý phim</h1>
    <h2 style='text-align: center; color: white;'>Hybrid (Content based + Collaborative)</h2>
""", unsafe_allow_html=True)

# Hiển thị tên người dùng ở góc phải nếu đã đăng nhập
if st.session_state.username:
    st.markdown(f"<h4 style='text-align: right; color: white;'>Xin chào, {st.session_state.username}</h4>", unsafe_allow_html=True)
movie_name = st.text_input("Tên phim")

# Đọc dữ liệu từ file new_movies.csv và rating.csv
@st.cache_data
def load_movie_data():
    try:
        new_movies = pd.read_csv("new_movies.csv")
        return new_movies[['movie_id', 'title', 'tags']]
    except FileNotFoundError:
        st.error("File 'new_movies.csv' không tìm thấy.")
        return pd.DataFrame(columns=['movie_id', 'title', 'tags'])
new_movies = load_movie_data()
@st.cache_data
def load_rating_data():
    try:
        new_ratings = pd.read_csv("ratings.csv")
        return new_ratings[['movie_id', 'user_id', 'rating', 'title']]
    except FileNotFoundError:
        st.error("File 'ratings.csv' không tìm thấy.")
        return pd.DataFrame(columns=['movie_id', 'user_id', 'rating', 'title'])
new_ratings = load_rating_data()

# Phương pháp 1: Dựa trên nội dung
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_movies['tags'])
content_similarity = cosine_similarity(vector)

def recommend_content_based(movie):
    try:
        movie_index = new_movies[new_movies['title'].str.lower() == movie.lower()].index[0]
    except IndexError:
        return []

    distances = content_similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[:5]
    recommend_movies = [new_movies.iloc[i[0]].title for i in movies_list]
    return recommend_movies

# Phương pháp 2: Dựa trên đánh giá
rating_matrix = new_ratings.pivot_table(index='user_id', columns='movie_id', values='rating')
scaler = StandardScaler()
rating_matrix_scaled = scaler.fit_transform(rating_matrix.fillna(0))
rating_similarity = cosine_similarity(rating_matrix_scaled.T)

def recommend_rating_based(movie_id):
    try:
        movie_index = rating_matrix.columns.get_loc(movie_id)
    except KeyError:
        return []

    sim_scores = sorted(list(enumerate(rating_similarity[movie_index])), key=lambda x: x[1], reverse=True)[:5]
    recommended_movie_ids = [rating_matrix.columns[i[0]] for i in sim_scores]
    return recommended_movie_ids

# Hàm lưu lịch sử tìm kiếm vào file CSV
def save_search_history(username, movie_name):
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, mode='r', newline='') as file:
            reader = csv.reader(file)
            existing_searches = []
            for row in reader:
                try:
                    # Lấy các phần tử từ danh sách row với slicing
                    row_data = row[:5]  # Slicing lấy các phần tử từ chỉ số 0 đến 4 (nếu có)

                    if len(row_data) > 4 and row_data[0] == username and row_data[2] == "searched" and row_data[
                        3] == "hybrid":
                        existing_searches.append(row_data[4])  # Truy cập phần tử thứ 5 nếu có
                except IndexError:
                    print("Lỗi: Một hàng không đủ phần tử.")
            if movie_name not in existing_searches:
                with open(USER_DB_FILE, mode='a', newline='') as file_append:
                    writer = csv.writer(file_append)
                    writer.writerow([username, "", "searched", "hybrid", movie_name])


# Đặt API Key TMDb của bạn ở đây
API_KEY = 'f87a3335b08012f880343b98c5d8374d'
BASE_URL = 'https://api.themoviedb.org/3'
POSTER_BASE_URL = 'https://image.tmdb.org/t/p/w500'


# Hàm lấy thông tin và poster phim từ TMDb
def fetch_movie_poster(movie_name):
    url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={movie_name}&language=en-US&page=1"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if data['results']:
            movie = data['results'][0]
            title = movie['title']
            poster_path = movie.get('poster_path', '')
            if poster_path:
                poster_url = f"{POSTER_BASE_URL}{poster_path}"
                return title, poster_url
    return None, None


# Hàm gợi ý phim
def hybrid_recommend(movie_name):
    content_recommendations = recommend_content_based(movie_name)
    content_ids = new_movies[new_movies['title'].isin(content_recommendations)]['movie_id'].tolist()

    rating_recommendations = []
    for movie_id in content_ids:
        rating_recommendations.extend(recommend_rating_based(movie_id))

    final_recommendations = list(dict.fromkeys(content_recommendations +
                                               [new_movies[new_movies['movie_id'] == mid].iloc[0].title
                                                for mid in rating_recommendations if
                                                mid in new_movies['movie_id'].values]))

    return final_recommendations[:10]


# Xử lý khi nhấn nút "Gợi ý"
if st.button("Gợi ý"):
    if st.session_state.username:
        if movie_name:
            save_search_history(st.session_state.username, movie_name)
            recommendations = hybrid_recommend(movie_name)

            if recommendations:
                st.write("Các phim gợi ý:")

                # Hiển thị các bộ phim gợi ý từ 1 đến 5
                cols_1_5 = st.columns(5)
                for i, col in enumerate(cols_1_5):
                    if i < len(recommendations):
                        movie_name = recommendations[i]
                        title, poster_url = fetch_movie_poster(movie_name)

                        if title and poster_url:
                            response = requests.get(poster_url)
                            if response.status_code == 200:
                                img = Image.open(BytesIO(response.content))
                                img = img.resize((200, 300))
                                col.image(img, use_container_width=True)
                            col.write(title)
                        else:
                            col.write(f"{movie_name} - Không tìm thấy poster")

                # Hiển thị các bộ phim gợi ý từ 6 đến 10
                if len(recommendations) > 5:
                    cols_6_10 = st.columns(5)
                    for i, col in enumerate(cols_6_10):
                        if i + 5 < len(recommendations):
                            movie_name = recommendations[i + 5]
                            title, poster_url = fetch_movie_poster(movie_name)

                            if title and poster_url:
                                response = requests.get(poster_url)
                                if response.status_code == 200:
                                    img = Image.open(BytesIO(response.content))
                                    img = img.resize((200, 300))
                                    col.image(img, use_container_width=True)
                                col.write(title)
                            else:
                                col.write(f"{movie_name} - Không tìm thấy poster")
            else:
                st.write("Không có phim gợi ý!")
        else:
            st.write("Vui lòng nhập tên phim!")
    else:
        st.write("Vui lòng đăng nhập để tìm kiếm!")

# Lịch sử tìm kiếm
if st.sidebar.button("Lịch sử tìm kiếm"):
    if st.session_state.username:
        search_history = []
        with open(USER_DB_FILE, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                try:
                    # Lấy các phần tử từ danh sách row với slicing (lấy tối đa 5 phần tử)
                    row_data = row[:5]

                    # Kiểm tra nếu row_data có đủ phần tử và thực hiện điều kiện
                    if len(row_data) > 2 and row_data[0] == st.session_state.username and row_data[2] == "searched":
                        search_history.append(row_data[4])  # Thêm phần tử thứ 5 nếu có
                except IndexError:
                    print("Lỗi: Một hàng không đủ phần tử.")

        search_history = list(set(search_history))
        if search_history:
            st.sidebar.subheader("Lịch sử tìm kiếm:")
            for movie in search_history:
                st.sidebar.write(f"- {movie}")
        else:
            st.sidebar.write("Chưa có lịch sử tìm kiếm.")
    else:
        st.sidebar.write("Vui lòng đăng nhập để xem lịch sử tìm kiếm.")


# Xóa lịch sử tìm kiếm
def clear_history_and_recommendations(username):
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, mode='r', newline='') as file:
            rows = list(csv.reader(file))

        # Dùng slicing để tránh lỗi IndexError khi row không có đủ phần tử
        rows = [
            row for row in rows if
            len(row) > 2 and row[0] == username and (row[2] == "searched" or row[2] == "recommended")
        ]

        # Ghi lại các dòng đã thay đổi vào file
        with open(USER_DB_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)



# Hiển thị danh sách phim nổi bật
def fetch_featured_movies():
    url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page=1"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("results", [])
    return []


st.subheader("Các bộ phim nổi bật")
cols = st.columns(5)

# Fetch and display movies
movies = fetch_featured_movies()

for i, col in enumerate(cols):
    if i < len(movies):
        movie = movies[i]
        title = movie["title"]
        poster_path = movie["poster_path"]

        if poster_path:
            poster_url = f"{POSTER_BASE_URL}{poster_path}"
            response = requests.get(poster_url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img = img.resize((400, 600))
                col.image(img, use_container_width=True)
        col.write(title)
    else:
        col.write("Không có phim")

