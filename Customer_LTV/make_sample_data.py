import pandas as pd
import numpy as np
import random
import os

current_path = os.getcwd()  # 현재 경로
parent_path = os.path.dirname(current_path)  # 상위 폴더 경로
data_dir = os.path.join(parent_path, "data")  # 상위 폴더의 data 폴더 경로

np.random.seed(42)
random.seed(42)

# 1. 사용자 정보 생성 및 고객군 세분화
num_users = 1000
user_ids = [f"user_{i:04d}" for i in range(num_users)]

# 고객군 비율 및 정의
segments = ["small", "medium", "large"]
segment_probs = [0.6, 0.3, 0.1]

user_segments = np.random.choice(segments, size=num_users, p=segment_probs)

signup_dates = pd.to_datetime(
    np.random.choice(pd.date_range("2022-01-01", "2024-06-01"), num_users)
)
referrers = np.random.choice(["organic", "referral", "social"], size=num_users)
kyc_levels = np.random.choice(["level_1", "level_2"], size=num_users, p=[0.7, 0.3])

users_df = pd.DataFrame(
    {
        "user_id": user_ids,
        "signup_date": signup_dates,
        "referrer": referrers,
        "kyc_level": kyc_levels,
        "segment": user_segments,
    }
)

# 2. 거래 기록 생성 (현실성 반영 거래 패턴)
transactions = []
assets = ["BTC", "ETH", "XRP", "SOL"]

for user_id, seg in zip(user_ids, user_segments):
    if seg == "small":
        num_tx = np.random.poisson(15)  # 거래 건수 많음
        avg_amount = 100000  # 10만 원 이하 평균 거래액
        amount_scale = 50000
    elif seg == "medium":
        num_tx = np.random.poisson(8)
        avg_amount = 1000000  # 100만 원 내외
        amount_scale = 300000
    else:  # large
        num_tx = np.random.poisson(3)  # 거래 건수 적음
        avg_amount = 10000000  # 1,000만 원 이상
        amount_scale = 3000000

    for _ in range(num_tx):
        trade_date = np.random.choice(pd.date_range("2022-01-01", "2024-06-01"))
        asset = np.random.choice(assets)
        trade_type = np.random.choice(["buy", "sell"])
        # 거래 금액은 평균을 중심으로 분포, 최소 10,000 이상 보장
        amount = max(10000, round(np.random.normal(avg_amount, amount_scale), 2))
        fee = round(amount * 0.0004, 2)  # 수수료 0.04% 고정
        transactions.append(
            [
                f"tx_{len(transactions):06d}",
                user_id,
                trade_date,
                asset,
                trade_type,
                amount,
                fee,
                "KRW",
            ]
        )

transactions_df = pd.DataFrame(
    transactions,
    columns=[
        "transaction_id",
        "user_id",
        "trade_datetime",
        "asset",
        "trade_type",
        "trade_amount",
        "fee_amount",
        "fee_asset",
    ],
)

# 3. 자산 스냅샷 생성 (고객군별 자산가치 차별화)
snapshots = []
snapshot_dates = pd.date_range(start="2022-01-01", end="2024-06-01", freq="ME")

for user_id, seg in zip(user_ids, user_segments):
    if seg == "small":
        base_value = np.random.uniform(500000, 3000000)  # 50만~300만 원
    elif seg == "medium":
        base_value = np.random.uniform(3000000, 15000000)  # 300만~1,500만 원
    else:
        base_value = np.random.uniform(15000000, 100000000)  # 1,500만~1억 원 이상

    signup_date = users_df.loc[users_df.user_id == user_id, "signup_date"].values[0]
    signup_date = pd.to_datetime(signup_date)

    for snap_date in snapshot_dates:
        if snap_date < signup_date:
            continue
        # 변동성 ±20% 반영
        value = base_value * (0.8 + np.random.rand() * 0.4)
        snapshots.append([snap_date, user_id, round(value, 2)])

snapshots_df = pd.DataFrame(
    snapshots, columns=["snapshot_date", "user_id", "total_asset_value"]
)

# 결과 샘플 출력
users_df.to_csv(f"{data_dir}/users_df.csv")
transactions_df.to_csv(f"{data_dir}/transactions_df.csv")
snapshots_df.to_csv(f"{data_dir}/snapshots_df.csv")
