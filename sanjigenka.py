# ==========================================
# 1. 必要なライブラリのインストール
# ==========================================
!pip install -q sentence-transformers plotly umap-learn fugashi ipadic

import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import umap.umap_ as umap

# ==========================================
# 2. データの読み込み
# ==========================================
# さっき作ったCSVファイルを読み込みます
filename = "movie_data_with_text.csv"
df = pd.read_csv(filename)

print(f"データ読み込み完了: {len(df)}件")

# ==========================================
# 3. AIによるベクトル化 
# ==========================================
print("AIモデルをロード中... (初回は少し時間がかかります)")
# 日本語に強いモデル
model = SentenceTransformer('oshizo/sbert-jsnli-luke-japanese-base-lite')

print("テキストをベクトル化中...")
# 「あらすじ」をベクトルに変換
embeddings = model.encode(df['text'].fillna("").tolist(), show_progress_bar=True)

# ==========================================
# 4. 次元削減
# ==========================================
# ここではPCAではなく、よりクラスターを作りやすいUMAPを使用
reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=5, min_dist=0.3)
coords = reducer.fit_transform(embeddings)

# データフレームに座標を追加
df['x'] = coords[:, 0]
df['y'] = coords[:, 1]
df['z'] = coords[:, 2]

# ==========================================
# 5. インタラクティブな可視化
# ==========================================
fig = px.scatter_3d(
    df,
    x='x', y='y', z='z',
    color='category',       # カテゴリごとに色分け
    text='title',           # 点の横にタイトルを表示
    hover_data=['text'],    # マウスオーバーであらすじを表示
    title='映画テイスト3Dマップ (AI分析版)',
    opacity=0.8,
    size_max=10
)

# 視認性
fig.update_traces(textposition='top center')
fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=30),
    legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1)
)

fig.show()

# HTMLファイルとして保存
fig.write_html("movie_taste_map.html")
print("movie_taste_map.html を保存しました。ダウンロードしてブラウザで開けます。")
from google.colab import files
files.download("movie_taste_map.html")
