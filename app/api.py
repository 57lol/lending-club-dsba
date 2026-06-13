# -*- coding: utf-8 -*-
"""
REST API к датасету Lending Club (FastAPI).

Эндпоинты:
  GET  /                  — информация об API
  GET  /loans             — выборка кредитов с фильтрами и пагинацией (>=2 аргумента)
  GET  /stats             — агрегированная статистика по группам (>=2 аргумента)
  POST /loans             — создание новой записи кредита
  GET  /loans/custom      — список созданных через POST записей

Запуск:  uvicorn app.api:app --reload      (из корня проекта)
Документация Swagger:     http://127.0.0.1:8000/docs
"""
from typing import Optional, List, Literal
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
import math

from app.data import load_clean_data, GRADES, TERMS

app = FastAPI(
    title="Lending Club API",
    description="REST API к датасету Lending Club: фильтрация, статистика и создание записей.",
    version="1.0.0",
)

DATA = load_clean_data()                     # чистый датафрейм (кэшируется)
CUSTOM_LOANS: List[dict] = []                # записи, созданные через POST

# колонки, которые отдаём наружу (без служебных дат)
OUT_COLS = ["loan_amnt", "term_months", "int_rate", "installment", "grade", "sub_grade",
            "annual_inc", "dti", "purpose", "home_ownership", "loan_status", "is_default"]


def _clean_records(df) -> List[dict]:
    """DataFrame -> список словарей с JSON-безопасными значениями."""
    recs = df[OUT_COLS].to_dict(orient="records")
    for r in recs:
        for k, v in r.items():
            if isinstance(v, float) and math.isnan(v):
                r[k] = None
    return recs


class LoanIn(BaseModel):
    """Модель для создания новой записи кредита (POST)."""
    loan_amnt: float = Field(..., gt=0, example=12000, description="Сумма кредита, $")
    term_months: Literal[36, 60] = Field(36, description="Срок кредита, мес.")
    int_rate: float = Field(..., gt=0, lt=100, example=13.5, description="Ставка, %")
    grade: str = Field(..., pattern="^[A-G]$", example="B", description="Грейд A–G")
    annual_inc: float = Field(..., gt=0, example=65000, description="Годовой доход, $")
    dti: float = Field(0, ge=0, le=60, example=17.0, description="Долговая нагрузка, %")
    purpose: str = Field("debt_consolidation", example="debt_consolidation",
                         description="Цель кредита")
    home_ownership: str = Field("RENT", example="RENT", description="Тип жилья")


class LoanOut(LoanIn):
    id: int
    installment: float = Field(..., description="Расчётный ежемесячный платёж, $")
    overpay_ratio: float = Field(..., description="Переплата в долях от тела кредита")


@app.get("/")
def root():
    return {
        "name": "Lending Club API",
        "rows_in_dataset": int(len(DATA)),
        "endpoints": {
            "GET /loans": "фильтр + пагинация (grade, term, purpose, min/max суммы и ставки, limit, offset)",
            "GET /stats": "агрегаты по группам (group_by, metric)",
            "POST /loans": "создать новую запись кредита",
            "GET /loans/custom": "созданные записи",
            "docs": "/docs",
        },
    }


@app.get("/loans")
def get_loans(
    grade: Optional[str] = Query(None, pattern="^[A-G]$", description="Фильтр по грейду A–G"),
    term: Optional[int] = Query(None, description="Срок: 36 или 60"),
    purpose: Optional[str] = Query(None, description="Цель кредита (подстрока)"),
    min_amnt: Optional[float] = Query(None, ge=0, description="Минимальная сумма, $"),
    max_amnt: Optional[float] = Query(None, ge=0, description="Максимальная сумма, $"),
    max_int_rate: Optional[float] = Query(None, gt=0, description="Максимальная ставка, %"),
    only_default: Optional[bool] = Query(None, description="Только дефолтные (true) / погашенные (false)"),
    limit: int = Query(20, ge=1, le=200, description="Сколько записей вернуть"),
    offset: int = Query(0, ge=0, description="Сдвиг для пагинации"),
):
    """Выборка кредитов с фильтрами и пагинацией. Минимум 2 аргумента (limit/offset + фильтры)."""
    df = DATA
    if grade is not None:
        df = df[df["grade"] == grade]
    if term is not None:
        if term not in TERMS:
            raise HTTPException(400, f"term должен быть одним из {TERMS}")
        df = df[df["term_months"] == term]
    if purpose is not None:
        df = df[df["purpose"].str.contains(purpose, case=False, na=False)]
    if min_amnt is not None:
        df = df[df["loan_amnt"] >= min_amnt]
    if max_amnt is not None:
        df = df[df["loan_amnt"] <= max_amnt]
    if max_int_rate is not None:
        df = df[df["int_rate"] <= max_int_rate]
    if only_default is not None:
        df = df[df["is_default"] == (1 if only_default else 0)]

    total = int(len(df))
    page = df.iloc[offset: offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "count": int(len(page)),
        "items": _clean_records(page),
    }


@app.get("/stats")
def get_stats(
    group_by: Literal["grade", "purpose", "term_months", "home_ownership"] = Query(
        "grade", description="Поле для группировки"),
    metric: Literal["default_rate", "avg_int_rate", "avg_loan_amnt", "count"] = Query(
        "default_rate", description="Метрика"),
):
    """Агрегированная статистика по группам (>=2 аргумента: group_by + metric)."""
    g = DATA.groupby(group_by)
    if metric == "default_rate":
        res = (g["is_default"].mean() * 100).round(2)
    elif metric == "avg_int_rate":
        res = g["int_rate"].mean().round(2)
    elif metric == "avg_loan_amnt":
        res = g["loan_amnt"].mean().round(2)
    else:  # count
        res = g.size()
    res = res.sort_values(ascending=False)
    return {"group_by": group_by, "metric": metric,
            "result": {str(k): float(v) for k, v in res.items()}}


@app.post("/loans", response_model=LoanOut, status_code=201)
def create_loan(loan: LoanIn):
    """Создаёт новую запись кредита: считает платёж и переплату, сохраняет в памяти."""
    r = loan.int_rate / 100 / 12                       # месячная ставка
    n = loan.term_months
    # аннуитетный платёж
    if r > 0:
        installment = loan.loan_amnt * r * (1 + r) ** n / ((1 + r) ** n - 1)
    else:
        installment = loan.loan_amnt / n
    installment = round(installment, 2)
    total_payment = installment * n
    overpay_ratio = round((total_payment - loan.loan_amnt) / loan.loan_amnt, 3)

    rec = LoanOut(id=len(CUSTOM_LOANS) + 1, installment=installment,
                  overpay_ratio=overpay_ratio, **loan.model_dump())
    CUSTOM_LOANS.append(rec.model_dump())
    return rec


@app.get("/loans/custom")
def list_custom():
    """Все записи, созданные через POST /loans."""
    return {"count": len(CUSTOM_LOANS), "items": CUSTOM_LOANS}
