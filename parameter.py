import streamlit as st

def parameter_selection(selected_model):
    """選択されたモデルに基づいてパラメータ選択画面を表示する関数

    Args:
        selected_model (str): 選択されたモデルの名前 ("tree", "rf", "lightgbm")

    Returns:
        dict: 選択されたパラメータ
    """
    model_options = {
        "tree": "決定木",
        "rf": "ランダムフォレスト",
        "lgb": "LightGBM",
    }
    
    params = {}
    
    # サイドバーにパラメータ設定を表示
    if selected_model != "knn" and selected_model != "lda":
        st.subheader(f"{model_options[selected_model]}のパラメータ")

    if selected_model == "tree":
        # 決定木のパラメータ選択 
        criterion_options = ["gini", "entropy"]
        criterion = st.selectbox("criterion", criterion_options)
        max_depth = st.slider("max_depth", 1, 10, 5, 1)
        params["criterion"] = criterion
        params["max_depth"] = max_depth

    elif selected_model == "rf":
        # ランダムフォレストのパラメータ選択
        n_estimators = st.slider("n_estimators (決定木の数)", 10, 200, 100, 10)
        max_depth = st.slider("max_depth (木の最大深さ)", 1, 10, 5, 1)
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth

    elif selected_model == "lgb":
        # LightGBMのパラメータ選択
        n_estimators = st.slider("n_estimators (決定木の数)", 10, 200, 100, 10)
        learning_rate = st.slider("learning_rate (学習率)", 0.01, 0.5, 0.1, 0.01)
        max_depth = st.slider("max_depth (木の最大深さ)", 1, 10, 3, 1)
        params["n_estimators"] = n_estimators
        params["learning_rate"] = learning_rate
        params["max_depth"] = max_depth

    else:
        pass
    
    return params