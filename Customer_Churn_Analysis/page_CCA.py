import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os


# st.set_page_config(
#     layout="wide",  #  화면을 넓게 설정
# )
def run():
    plt.rcParams["font.family"] = "Malgun Gothic"  # 윈도우
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({"font.size": 10})

    current_path = os.getcwd()  # 현재 경로
    data_dir = os.path.join(current_path, "data")
    # -- 1. 데이터 로드 (churn_df) --
    # 예) CSV 불러오기 (이미 churn_df 생성 후 저장했다고 가정)
    churn_df = pd.read_csv(
        f"{data_dir}/customer_churn_data.csv", parse_dates=["last_login_date"]
    )

    # -- 2. 기본 통계 및 시각화 --

    st.title("이탈 예측 모델링: 데이터 탐색 및 모델 성능 비교")

    st.markdown(
        """
    ### 0. 데이터 개요
    본 분석은 고객의 **이탈 여부(Churn)** 를 예측하기 위해 아래 변수들을 활용했습니다.

    - `user_id`: 고객 식별자
    - `last_login_date`: 고객의 마지막 로그인 날짜
    - `num_trades_last_30d`: 최근 30일 내 거래 횟수
    - `num_trades_last_90d`: 최근 90일 내 거래 횟수
    - `num_inquiries_last_30d`: 최근 30일 내 고객 문의 횟수
    - `marketing_opt_in`: 마케팅 수신 동의 여부 (True/False)
    - `days_since_last_login`: 마지막 로그인 이후 지난 일수
    - `churn`: 이탈 여부 (1: 이탈, 0: 유지)

    **기준일(reference_date)** 은 2024-06-01로 설정하여, 이 날짜를 기준으로 데이터를 집계하였습니다.
    """
    )

    st.markdown("### 1. 이탈(churn) 데이터 기초 통계")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=churn_df, x="churn", ax=ax)
    ax.set_title("이탈 여부 분포 (0=유지, 1=이탈)")
    st.dataframe(churn_df.describe())
    col99, col98 = st.columns((3, 3))
    with col99:
        st.pyplot(fig)
    with col98:
        st.markdown("**설명**")
        st.write("- 학습에 사용되는 라벨 분포의 균형을 확인합니다다")
        st.write("- 2년 이상의 기간을 확인해 이탈자가 더 많이 나오는 것을 확인")

    st.markdown("#### 변수별 분포 및 이탈과의 관계")

    # 1. 최근 30일 거래 횟수
    col1, col2 = st.columns([1, 1])
    with col1:
        fig1, ax1 = plt.subplots()
        sns.histplot(
            churn_df["num_trades_last_30d"], bins=30, kde=False, ax=ax1, color="skyblue"
        )
        ax1.set_title("최근 30일 거래 횟수 분포")
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        st.pyplot(fig1)
    with col2:
        st.markdown(
            """
        **📌 지표 설명**
        - `num_trades_last_30d`: 사용자가 최근 30일간 수행한 거래 횟수입니다.
        - 거래 횟수가 적은 사용자는 이탈 가능성이 높게 나타날 수 있습니다.
        - 이 지표는 사용자 **단기 활동성**을 나타내며, 이탈 예측에 중요한 변수입니다.
        """
        )

    # 2. 마지막 로그인 이후 일수
    col3, col4 = st.columns([1, 1])
    with col3:
        fig2, ax2 = plt.subplots()
        sns.histplot(
            churn_df["days_since_last_login"],
            bins=30,
            kde=False,
            ax=ax2,
            color="salmon",
        )
        ax2.set_title("마지막 로그인 이후 경과 일수 분포")
        st.pyplot(fig2)
    with col4:
        st.markdown(
            """
        **📌 지표 설명**
        - `days_since_last_login`: 현재 기준 마지막 로그인 후 경과 일수입니다.
        - 값이 클수록 장기 미접속 사용자일 가능성이 높습니다.
        - 60일 이상 미접속 시 이탈로 간주하였습니다.
        """
        )

    # 3. 마케팅 수신 여부 vs 이탈
    col5, col6 = st.columns([1, 1])
    with col5:
        fig3, ax3 = plt.subplots()
        sns.countplot(
            data=churn_df, x="marketing_opt_in", hue="churn", ax=ax3, palette="Set2"
        )
        ax3.set_title("마케팅 수신 동의 여부와 이탈 여부")
        st.pyplot(fig3)
    with col6:
        st.markdown(
            """
        **📌 지표 설명**
        - 마케팅 수신 동의 여부(`marketing_opt_in`)에 따라 이탈율 차이를 확인합니다.
        - 수신 동의자와 비동의자 간의 이탈률 차이는 **리텐션 전략** 수립에 유의미합니다.
        """
        )

    # 4. 최근 30일 문의 횟수
    col7, col8 = st.columns([1, 1])
    with col7:
        fig4, ax4 = plt.subplots()
        sns.histplot(
            churn_df["num_inquiries_last_30d"],
            bins=20,
            kde=False,
            ax=ax4,
            color="lightgreen",
        )
        ax4.set_title("최근 30일 문의 횟수 분포")
        ax4.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        st.pyplot(fig4)
    with col8:
        st.markdown(
            """
        **📌 지표 설명**
        - 고객 문의 횟수는 **이탈 가능성의 사전 징후**일 수 있습니다.
        - 불만 또는 이슈를 겪은 유저는 문의 후 이탈할 가능성이 높습니다.
        - 높은 문의 빈도는 **불편한 사용자 경험**을 나타낼 수 있습니다.
        """
        )

    st.markdown("---")

    st.markdown("### 2. 모델 학습 및 평가")

    # 특성 및 타겟 분리
    feature_cols = [
        "days_since_last_login",
        "num_trades_last_30d",
        "num_trades_last_90d",
        "num_inquiries_last_30d",
        "marketing_opt_in",
    ]
    X = churn_df[feature_cols].copy()

    # 마케팅 수신 여부 True/False -> 1/0 변환
    X["marketing_opt_in"] = X["marketing_opt_in"].astype(int)

    y = churn_df["churn"]

    # 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42, test_size=0.2
    )

    # -- 3-1. 로지스틱 회귀 --

    st.markdown("#### 2.1 로지스틱 회귀")

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]

    st.write("분류 리포트:")
    st.dataframe(
        pd.DataFrame(
            classification_report(y_test, y_pred_lr, output_dict=True)
        ).transpose()
    )

    roc_auc_lr = roc_auc_score(y_test, y_proba_lr)
    st.write(f"ROC AUC Score: {roc_auc_lr:.4f}")

    # -- 3-2. XGBoost --

    st.markdown("#### 2.2 XGBoost")

    xgb_clf = xgb.XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", random_state=42
    )
    xgb_clf.fit(X_train, y_train)
    y_pred_xgb = xgb_clf.predict(X_test)
    y_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]

    st.write("분류 리포트:")
    st.dataframe(
        pd.DataFrame(
            classification_report(y_test, y_pred_xgb, output_dict=True)
        ).transpose()
    )
    roc_auc_xgb = roc_auc_score(y_test, y_proba_xgb)
    st.write(f"ROC AUC Score: {roc_auc_xgb:.4f}")

    # -- 3-3. LSTM (시계열 특성 단순화 예시) --

    st.markdown("#### 2.3 LSTM (단순 시계열 모델)")

    # LSTM은 보통 시퀀스 데이터가 필요하므로,
    # 여기서는 'days_since_last_login' ~ 'num_inquiries_last_30d' 특성들을 1시계열 시퀀스처럼 가정하여 훈련하는 예시

    # 단순화를 위해 동일 feature 컬럼들을 1개 timestep으로 reshape (실제로 시계열 데이터면 여러 timestep 필요)
    X_train_lstm = np.expand_dims(
        X_train.values, axis=1
    )  # shape: (samples, timesteps=1, features)
    X_test_lstm = np.expand_dims(X_test.values, axis=1)

    from tensorflow.keras.callbacks import EarlyStopping

    model = Sequential(
        [
            LSTM(
                32,
                input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]),
                activation="relu",
            ),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    early_stop = EarlyStopping(monitor="val_loss", patience=3)

    history = model.fit(
        X_train_lstm,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0,
    )

    # 예측 및 평가
    y_proba_lstm = model.predict(X_test_lstm).flatten()
    y_pred_lstm = (y_proba_lstm >= 0.5).astype(int)

    from sklearn.metrics import accuracy_score

    st.write(f"Accuracy: {accuracy_score(y_test, y_pred_lstm):.4f}")

    roc_auc_lstm = roc_auc_score(y_test, y_proba_lstm)
    st.write(f"ROC AUC Score: {roc_auc_lstm:.4f}")

    # -- 모델 성능 요약 --
    st.header("3. 모델 성능 비교")

    performance_df = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "XGBoost", "LSTM"],
            "ROC AUC": [roc_auc_lr, roc_auc_xgb, roc_auc_lstm],
        }
    )

    st.dataframe(performance_df)

    st.markdown("---")
    st.header(" 최종 결론 및 요약")

    st.markdown(
        """
        이번 프로젝트에서는 고객 데이터를 기반으로 **이탈 예측(Churn Prediction)**을 위한 세 가지 모델을 적용했습니다:

        1. **로지스틱 회귀 (Logistic Regression)**  
        - 해석이 쉽고, 주요 변수의 영향력을 파악하는 데 유용  
        - 기본적인 이탈 패턴 이해에 적합

        2. **XGBoost (Gradient Boosting)**  
        - 비선형 관계와 복잡한 상호작용을 잘 포착  
        - 비교적 높은 예측 성능을 보여주어 실제 서비스 적용 가능성 큼

        3. **LSTM (시계열 딥러닝)**  
        - 시간에 따른 고객 행동 변화를 반영하여 시계열 특성 활용  
        - 장기적인 고객 활동 추이를 기반으로 보다 정교한 예측 가능

        ### 활용 방안
        - 이탈 가능성이 높은 고객을 조기에 식별하여 타겟 마케팅 및 맞춤형 프로모션을 진행  
        - 마지막 로그인 일수, 최근 거래 빈도, 문의 횟수, 마케팅 수신 여부 등 핵심 변수에 집중하여 고객 유지 전략 수립  
        - 실제 서비스에서는 지속적인 데이터 업데이트와 모델 재학습을 통해 예측 정확도를 높임

        ### 다음 단계 제안
        - 고객 세분화별 이탈 예측 모델 성능 비교 및 최적화  
        - 이상 거래 탐지 모델과 결합한 통합 리스크 관리 체계 구축  
        - 고객 LTV 분석과 연계한 맞춤형 유치/재유치 전략 강화  

        본 프로젝트 결과는 고객 이탈을 줄이고, 장기적 고객 가치를 극대화하는 데 중요한 기초 자료로 활용될 수 있습니다.
        """
    )
