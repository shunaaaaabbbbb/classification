import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score
import plotly.graph_objects as go

def display_result(train_score, test_score, y_test, y_pred):
    """分類結果を表示する関数

    Args:
        train_score (float): 訓練データの精度
        test_score (float): テストデータの精度
        y_test (pd.Series): テストデータの実際の値
        y_pred (np.array): テストデータの予測値
    """

    col1, col2 = st.columns([1,3])  # 列の幅を調整

    with col1:
        st.write("")
        st.title("")
        st.metric("Accuracy（訓練データ）", f"{train_score:.3f}")  # metricを使って表示
        st.metric("Accuracy（テストデータ）", f"{test_score:.3f}")

        f1 = f1_score(y_test, y_pred, average="macro")
        st.metric("F1 score（テストデータ）", f"{f1:.3f}")  
    

    with col2:
        # plotlyを使ってConfusion Matrixを作成
        cm = confusion_matrix(y_test, y_pred)

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f"予測ラベル: {i}" for i in range(cm.shape[1])],
            y=[f"正解ラベル: {i}" for i in range(cm.shape[0])],
            #colorscale="blugrn",
            colorscale="brwnyl",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "white"},  # テキストのフォント設定
            hovertemplate="<b><span style='font-size:15px; color:black;'>%{x}</span></b><br>"  # 予測ラベル
                 "<b><span style='font-size:15px; color:black;'>%{y}</span></b><br>"  # 正解ラベル
                 "<b><span style='font-size:15px; color:black;'>予測数: %{z}</span></b>"  # 総数
                 "<extra></extra>",  # hovertemplate をカスタマイズ
        ))

        fig.update_layout(
            yaxis_autorange="reversed",
            title_text="<b>Confusion Matrix</b>",
            title_x=0.5,
            title_font_size=24,
            font_family="Arial",
            font_color="black",
            xaxis_title="<b>予測ラベル</b>",
            yaxis_title="<b>正解ラベル</b>",
            xaxis_title_font_size=18,
            yaxis_title_font_size=18,
            xaxis_tickfont_size=14,
            yaxis_tickfont_size=14,
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor="lightgray",
            xaxis=dict(showgrid=False),  # x軸のグリッド線を非表示
            yaxis=dict(showgrid=False),  # y軸のグリッド線を非表示
        )

        st.plotly_chart(fig)