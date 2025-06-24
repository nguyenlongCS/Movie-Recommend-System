#streamlit run content_based.py

import streamlit as st
import pandas as pd
import csv
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Cấu hình chung
st.set_page_config(page_title="Gợi ý nội dung", layout="wide", initial_sidebar_state="expanded")

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
    <h2 style='text-align: center; color: white;'>Content Based Filtering</h2>
""", unsafe_allow_html=True)

# Hiển thị tên người dùng ở góc phải nếu đã đăng nhập
if st.session_state.username:
    st.markdown(f"<h4 style='text-align: right; color: white;'>Xin chào, {st.session_state.username}</h4>", unsafe_allow_html=True)
movie_name = st.text_input("Tên phim")

# Đọc dữ liệu từ file new_movies.csv
@st.cache_data
def load_movie_data():
    try:
        new_movies = pd.read_csv("new_movies.csv")
        return new_movies[['movie_id', 'title', 'tags']]
    except FileNotFoundError:
        st.error("File 'new_movies.csv' không tìm thấy.")
        return pd.DataFrame(columns=['movie_id', 'title', 'tags'])

new_movies = load_movie_data()

# Tính toán ma trận độ tương đồng cosine dựa trên nội dung mô tả phim
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_movies['tags'])
similarity = cosine_similarity(vector)

# Hàm lấy gợi ý phim
def recommend(movie):
    try:
        movie_index = new_movies[new_movies['title'].str.lower() == movie.lower()].index[0]
    except IndexError:
        return ["Phim không có trong dữ liệu!"]

    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[:5]

    recommend_movies = []
    for i in movies_list:
        recommend_movies.append(new_movies.iloc[i[0]].title)
    return recommend_movies

# Hàm lưu lịch sử tìm kiếm vào file CSV
def save_search_history(username, movie_name):
    if os.path.exists(USER_DB_FILE):
        # Kiểm tra nếu tên phim chưa có trong lịch sử tìm kiếm của người dùng
        with open(USER_DB_FILE, mode='r', newline='') as file:
            reader = csv.reader(file)
            existing_searches = [row[4] for row in reader if row[0] == username and row[2] == "searched" and row[3] == "content"]
            if movie_name not in existing_searches:
                # Nếu không có trong lịch sử, thêm vào lịch sử tìm kiếm
                with open(USER_DB_FILE, mode='a', newline='') as file_append:
                    writer = csv.writer(file_append)
                    writer.writerow([username, "", "searched", "content", movie_name])

# Xử lý khi nhấn nút "Gợi ý"
if st.button("Gợi ý"):
    if st.session_state.username:
        if movie_name:
            # Lưu lịch sử tìm kiếm vào file CSV
            save_search_history(st.session_state.username, movie_name)

            recommendations = recommend(movie_name)
            if recommendations[0] != "Phim không có trong dữ liệu!":
                st.write("Các phim gợi ý:")
                for i, rec in enumerate(recommendations, start=1):
                    st.write(f"{i}. {rec}")
            else:
                st.write(recommendations[0])
        else:
            st.write("Vui lòng nhập tên phim!")
    else:
        st.write("Vui lòng đăng nhập để tìm kiếm!")

# Lịch sử tìm kiếm (Button)
if st.sidebar.button("Lịch sử tìm kiếm"):
    if st.session_state.username:
        # Đọc dữ liệu từ file để lấy lịch sử tìm kiếm của người dùng
        search_history = []
        with open(USER_DB_FILE, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == st.session_state.username and row[2] == "searched":
                    search_history.append(row[4])

        # Loại bỏ các tìm kiếm trùng lặp
        search_history = list(set(search_history))

        if search_history:
            st.sidebar.subheader("Lịch sử tìm kiếm:")
            for movie in search_history:
                st.sidebar.write(f"- {movie}")
        else:
            st.sidebar.write("Chưa có lịch sử tìm kiếm.")
    else:
        st.sidebar.write("Vui lòng đăng nhập để xem lịch sử tìm kiếm.")


# Xóa lịch sử tìm kiếm và đề xuất
def clear_history_and_recommendations(username):
    if os.path.exists(USER_DB_FILE):
        # Đọc tất cả các dòng trong file
        with open(USER_DB_FILE, mode='r', newline='') as file:
            rows = list(csv.reader(file))

        # Lọc các dòng không phải là lịch sử tìm kiếm và đề xuất của người dùng
        rows = [row for row in rows if not (row[0] == username and (row[2] == "searched" or row[2] == "recommended"))]

        # Ghi lại các dòng còn lại vào file CSV
        with open(USER_DB_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

if st.sidebar.button("Xóa lịch sử tìm kiếm"):
    if st.session_state.username:
        clear_history_and_recommendations(st.session_state.username)
        st.sidebar.success("Đã xóa lịch sử tìm kiếm và đề xuất.")

# Hiển thị danh sách phim nổi bật
import requests
from PIL import Image
from io import BytesIO
import streamlit as st

# TMDb API Key
API_KEY = "f87a3335b08012f880343b98c5d8374d"
BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"


# Fetch featured movies data
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




