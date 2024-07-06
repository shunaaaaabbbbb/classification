import streamlit as st

def page_howto():


    st.title("このサイトの使い方")
    st.subheader("分類したいデータを選択してください。")
    st.image("image/data.png")
    st.title("\u2193")
    st.subheader("軸を選択してデータの散布図を見てみましょう。")
    st.image("image/ax.png")
    st.title("\u2193")
    st.subheader("どのモデルを使うかを選択してください。")
    st.image("image/model.png")
    st.title("\u2193")
    st.subheader("パラメータを選択してください。（線形判別とk近傍法に関してはパラメータ選択はないです。）")
    st.image("image/parameter.png")
    st.title("\u2193")
    st.subheader("計算実行ボタンを押すと分類結果としてAccuracy, F1 score, Confusion Matrixが表示されます。")
    st.image("image/result.png")
    st.title("\u2193")
    st.subheader("Confusion Matrixにカーソルを当てると予測ラベルと正解ラベルと予測数が表示されます。")
    st.image("image/cursor.png")
