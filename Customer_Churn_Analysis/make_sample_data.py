import pandas as pd
import numpy as np
import os

current_path = os.getcwd()  # 현재 경로
parent_path = os.path.dirname(current_path)  # 상위 폴더 경로

data_dir = os.path.join(parent_path, "data")  # 상위 폴더의 data 폴더 경로

users_df = pd.read_csv(os.path.join(data_dir, "users_df.csv"))
transactions_df = pd.read_csv(os.path.join(data_dir, "transactions_df.csv"))

# 기준일 설정 (예: 2024-06-01)
reference_date = pd.to_datetime("2024-06-01")

# 1) 마지막 로그인 날짜 생성 (가입일 이후 랜덤 활동 일수)
users_df["signup_date"] = pd.to_datetime(users_df["signup_date"])

users_df["last_login_date"] = users_df["signup_date"] + pd.to_timedelta(
    np.random.randint(0, 600, size=len(users_df)), unit="D"
)
users_df.loc[users_df["last_login_date"] > reference_date, "last_login_date"] = (
    reference_date
)

# 2) 거래 건수 집계 (최근 30일, 90일)
transactions_df["trade_datetime"] = pd.to_datetime(transactions_df["trade_datetime"])
trade_30d = transactions_df[
    (transactions_df["trade_datetime"] >= reference_date - pd.Timedelta(days=30))
    & (transactions_df["trade_datetime"] <= reference_date)
]
trade_90d = transactions_df[
    (transactions_df["trade_datetime"] >= reference_date - pd.Timedelta(days=90))
    & (transactions_df["trade_datetime"] <= reference_date)
]

num_trades_30d = trade_30d.groupby("user_id").size().rename("num_trades_last_30d")
num_trades_90d = trade_90d.groupby("user_id").size().rename("num_trades_last_90d")

# 3) 문의 데이터 (샘플로 생성)
np.random.seed(42)
inquiries = []
for user in users_df["user_id"]:
    inquiries.append(
        {
            "user_id": user,
            "num_inquiries_last_30d": np.random.poisson(0.2),  # 평균 0.2건 문의
        }
    )
inquiries_df = pd.DataFrame(inquiries)

# 4) 마케팅 수신 여부 (랜덤)
users_df["marketing_opt_in"] = np.random.choice(
    [True, False], size=len(users_df), p=[0.6, 0.4]
)

# 5) 통합 및 결측 처리
churn_df = users_df[["user_id", "last_login_date", "marketing_opt_in"]].copy()
churn_df = (
    churn_df.merge(num_trades_30d, on="user_id", how="left")
    .merge(num_trades_90d, on="user_id", how="left")
    .merge(inquiries_df, on="user_id", how="left")
)
churn_df.fillna(
    {"num_trades_last_30d": 0, "num_trades_last_90d": 0, "num_inquiries_last_30d": 0},
    inplace=True,
)

# 6) 이탈 여부 라벨링
churn_df["days_since_last_login"] = (
    reference_date - churn_df["last_login_date"]
).dt.days

churn_prob = (
    0.6 * (churn_df["days_since_last_login"] > 60).astype(int)
    + 0.3 * (churn_df["num_trades_last_30d"] == 0).astype(int)
    + 0.1 * (churn_df["marketing_opt_in"] == False).astype(int)
)

# 약간의 확률적 요소 추가
churn_df["churn"] = np.random.binomial(n=1, p=np.clip(churn_prob, 0, 1))

print(churn_df.head())
# 7) 최종 데이터 저장
churn_df.to_csv("../data/customer_churn_data.csv", index=False)
