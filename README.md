# movie_mapping - 映画テイストの3次元マッピング

AI（SentenceTransformer）と次元削減アルゴリズム（UMAP）を用いて、映画のあらすじテキストから「テイスト」を解析し、3次元空間にマッピング・可視化するプロジェクトです。

---

## 📂 ファイル構成

- **`sanjigen.py`**  
  映画データをAIでベクトル化し、UMAPで3次元に圧縮して Plotly で描画するメインの Python スクリプトです。
  
- **`movie_data_with_text.csv`**  
  分析対象となる映画のデータセットです。各映画のタイトル（`title`）、解説テキスト（`text`）、カテゴリ（`category`）が含まれています。
  
- **`movie_taste_map.html`**  
  `sanjigen.py` の実行によって書き出された 3D マップの HTML ファイルです。ブラウザで開くだけで、映画同士のテイストの近さを3次元空間上で回転・ズームしながら直感的に探索できます。

---

## 🛠️ セットアップと実行方法

### 1. 必要ライブラリのインストール
Python 環境（または Google Colab などのノートブック環境）で以下のライブラリをインストールします。

```bash
pip install sentence-transformers plotly umap-learn fugashi ipadic pandas
```

### 2. 実行手順

#### 💻 ローカルPC環境で実行する場合
1. `sanjigen.py` と `movie_data_with_text.csv` を同じフォルダに置きます。
2. スクリプトの最終行付近にある Google Colab 専用コード（`from google.colab import files...`）をコメントアウト（または削除）します。
3. 以下のコマンドを実行します。
   ```bash
   python sanjigen.py
   ```
4. 生成された `movie_taste_map.html` をブラウザで開いて確認します。

#### ☁️ Google Colab などのクラウド環境で実行する場合
1. `movie_data_with_text.csv` を Colab のファイルエリアにアップロードします。
2. セルにコードを貼り付けて実行します。
3. 実行完了後、生成された `movie_taste_map.html` が自動的にローカルPCにダウンロードされます。

---

## 🔬 技術仕様
- **テキストの埋め込み (Embedding):** 日本語に特化した軽量なSBERTモデル `oshizo/sbert-jsnli-luke-japanese-base-lite` を使用して、あらすじテキストをベクトル化しています。
- **次元削減:** 高次元の埋め込みベクトルを、テイストの類似性（局所的な関係性）を保ったまま 3次元に落とし込むために `UMAP` を使用しています。
- **可視化:** `Plotly Express (scatter_3d)` を使い、カテゴリでの色分け、タイストの3Dマッピング、およびマウスホバーによるあらすじ表示を実現しています。
