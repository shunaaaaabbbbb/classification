import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def select_data(data_options):
    st.subheader("データを選択する")
    selected_data = st.selectbox("分類したいデータの選択", list(data_options.keys()), format_func=lambda x: data_options[x])
    df_data = pd.read_csv(f"data/{selected_data}")
    return selected_data, df_data


def select_ax(df_data):
    """データフレームを受け取り、特徴量選択に応じて散布図を表示する関数

    Args:
        data (pd.DataFrame): 表示するデータフレーム
    """ 
    st.subheader("散布図を表示する")
    # 特徴量のリストを取得
    features = df_data.columns.tolist()
    features.remove("target")
    x_feature = st.selectbox("横軸を選択", features)
    y_feature = st.selectbox("縦軸を選択", features)
    return x_feature, y_feature


def plot_scatter(df_data, x_feature, y_feature):
    # 選択された特徴量で散布図を描画
    if x_feature and y_feature:  # 両方の特徴量が選択されている場合のみ実行
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_data, x=x_feature, y=y_feature, hue="target", ax=ax)
        st.pyplot(fig)