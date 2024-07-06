import streamlit as st

from classification import load_and_split_data, lda_classification, decision_tree_classification,random_forest_classification, knn_classification, lightgbm_classification
from display_result import display_result
from parameter import parameter_selection
from dataset import select_data, select_ax, plot_scatter


def page_main():
    st.title("色々なモデルで分類を試してみよう！")
    st.header("このwebアプリでは分類したいデータとモデルを選択したら分類をすることができます。")

    # データ選択
    data_options = {
        "data1.csv": "データ1（特徴量の数：2個、2値分類）",
        "data2.csv": "データ2（特徴量の数：3個、2値分類）",
        "data3.csv": "データ3（特徴量の数：4個、3値分類）",
        "data4.csv": "データ4（特徴量の数：5個、4値分類）",
    }
    col1,col2 = st.columns(2)
    with col1:
        selected_data, df_data = select_data(data_options)
        x_feature, y_feature = select_ax(df_data)
    with col2:
        plot_scatter(df_data, x_feature, y_feature)
   
    # データの読み込みと分割
    X_train, X_test, y_train, y_test = load_and_split_data(f"data/{selected_data}")

    # モデル選択
    model_options = {
        "lda": "線形判別（パラメータ選択なし）",
        "knn": "k近傍法（パラメータ選択なし）",
        "tree": "決定木",
        "rf": "ランダムフォレスト",
        "lgb": "lightgbm",
    }

    with st.sidebar:
        st.title("")
        st.title("モデル・パラメータの選択")
        selected_model = st.selectbox("モデルを選択してください", list(model_options.keys()), format_func=lambda x: model_options[x])
        params = parameter_selection(selected_model)

        # 計算実行ボタン
        if st.button("計算実行"):
            # 計算を実行し、結果をsession_stateに保存
            if selected_model == "lda":
                st.session_state.train_score, st.session_state.test_score, st.session_state.y_test, st.session_state.y_pred = lda_classification(X_train, X_test, y_train, y_test)
            elif selected_model == "knn":
                st.session_state.train_score, st.session_state.test_score, st.session_state.y_test, st.session_state.y_pred = knn_classification(X_train, X_test, y_train, y_test)
            elif selected_model == "tree":
                st.session_state.train_score, st.session_state.test_score, st.session_state.y_test, st.session_state.y_pred = decision_tree_classification(X_train, X_test, y_train, y_test, **params)
            elif selected_model == "rf":
                st.session_state.train_score, st.session_state.test_score, st.session_state.y_test, st.session_state.y_pred = random_forest_classification(X_train, X_test, y_train, y_test, **params)
            elif selected_model == "lgb":
                st.session_state.train_score, st.session_state.test_score, st.session_state.y_test, st.session_state.y_pred = lightgbm_classification(X_train, X_test, y_train, y_test, **params)

    # 結果の表示 (session_stateにデータがあれば表示)
    if "train_score" in st.session_state:
        display_result(st.session_state.train_score, st.session_state.test_score, st.session_state.y_test, st.session_state.y_pred)
        
    st.title("")
    st.write("※データ数は330で、そのうち80%を訓練データ、20%をテストデータに分けています。")