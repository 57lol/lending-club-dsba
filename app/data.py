# -*- coding: utf-8 -*-
"""
Общая загрузка и очистка данных Lending Club.
Повторяет логику предобработки из ноутбука; используется и FastAPI, и Streamlit.
"""
import os
import functools
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "..", "data", "lending_club_loan.csv")

EMP_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}


@functools.lru_cache(maxsize=1)
def load_clean_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Читает CSV, чистит данные и добавляет производные признаки. Кэшируется."""
    # не читаем тяжёлые свободно-текстовые колонки (их всё равно выкидываем) —
    # экономит память, важно для бесплатного облачного хостинга
    data = pd.read_csv(csv_path, usecols=lambda c: c not in ("emp_title", "title"))

    # текст -> числа
    data["term_months"] = data["term"].str.extract(r"(\d+)").astype(int)
    data["emp_length_num"] = data["emp_length"].str.strip().map(EMP_MAP)
    data["emp_length_num"] = data["emp_length_num"].fillna(data["emp_length_num"].median())
    data["state"] = data["address"].str.extract(r"([A-Z]{2})\s+\d{5}\s*$")
    data["is_default"] = (data["loan_status"] == "Charged Off").astype(int)

    # пропуски
    data["mort_acc"] = data["mort_acc"].fillna(data["mort_acc"].median())
    data["revol_util"] = data["revol_util"].fillna(data["revol_util"].median())
    data["pub_rec_bankruptcies"] = data["pub_rec_bankruptcies"].fillna(0)

    # выкидываем свободный текст и дубли полей, остаточные пропуски, аномалии
    data = data.drop(columns=["address", "term", "emp_length"])
    data = data.dropna().reset_index(drop=True)
    data = data[(data["dti"] <= 60) & (data["annual_inc"] > 0)].reset_index(drop=True)

    # производные признаки
    grade_order = sorted(data["grade"].unique())
    data["grade_num"] = data["grade"].map({g: i + 1 for i, g in enumerate(grade_order)})
    data["issue_date"] = pd.to_datetime(data["issue_d"], format="%b-%Y")
    data["total_payment"] = data["installment"] * data["term_months"]
    data["total_interest"] = data["total_payment"] - data["loan_amnt"]
    data["overpay_ratio"] = data["total_interest"] / data["loan_amnt"]
    data["payment_to_income"] = data["installment"] * 12 / data["annual_inc"]
    earliest = pd.to_datetime(data["earliest_cr_line"], format="%b-%Y")
    data["credit_history_years"] = ((data["issue_date"] - earliest).dt.days / 365.25).round(1)

    return data


# поля, которые отдаём/принимаем через API
NUMERIC_FIELDS = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "revol_util"]
GRADES = list("ABCDEFG")
TERMS = [36, 60]
