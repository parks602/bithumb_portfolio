import pandas as pd
import numpy as np
import os

current_path = os.getcwd()  # 현재 경로
parent_path = os.path.dirname(current_path)  # 상위 폴더 경로

data_dir = os.path.join(parent_path, "data")  # 상위 폴더의 data 폴더 경로
np.random.seed(42)
n_users = 1000
base_date = pd.to_datetime("2024-04-01")

user_ids = [f"user_{i:04d}" for i in range(n_users)]
signed_up_at = [
    base_date
    + pd.to_timedelta(np.random.randint(0, 30), unit="D")
    + pd.to_timedelta(np.random.randint(0, 24), unit="h")
    for _ in range(n_users)
]

# 각 단계에서 전환하지 않은 유저도 존재하게끔 확률 설정
kyc_offset_days = np.random.choice(
    [np.random.randint(0, 3), None], size=n_users, p=[0.85, 0.15]
)
deposit_offset_days = np.random.choice(
    [np.random.randint(0, 5), None], size=n_users, p=[0.7, 0.3]
)
first_trade_offset_days = np.random.choice(
    [np.random.randint(0, 2), None], size=n_users, p=[0.6, 0.4]
)
repeat_trade_offset_days = np.random.choice(
    [np.random.randint(1, 10), None], size=n_users, p=[0.5, 0.5]
)


def apply_offset(base_times, offsets):
    result = []
    for base, offset in zip(base_times, offsets):
        if offset is None:
            result.append(pd.NaT)
        else:
            dt = (
                base
                + pd.to_timedelta(offset, unit="D")
                + pd.to_timedelta(np.random.randint(0, 24), unit="h")
            )
            result.append(dt)
    return result


kyc_at = apply_offset(signed_up_at, kyc_offset_days)
deposit_at = apply_offset(kyc_at, deposit_offset_days)
first_trade_at = apply_offset(deposit_at, first_trade_offset_days)
repeat_trade_at = apply_offset(first_trade_at, repeat_trade_offset_days)

funnel_df = pd.DataFrame(
    {
        "user_id": user_ids,
        "signed_up_at": signed_up_at,
        "kyc_at": kyc_at,
        "deposit_at": deposit_at,
        "first_trade_at": first_trade_at,
        "repeat_trade_at": repeat_trade_at,
    }
)

print(funnel_df.head())
funnel_df.to_csv(f"{data_dir}/funnel_data.csv", index=False)
