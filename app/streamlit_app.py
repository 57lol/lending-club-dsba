# -*- coding: utf-8 -*-
"""
Веб-версия отчёта по датасету Lending Club (Streamlit).
Эквивалент Jupyter-ноутбука: все разделы, графики и выводы — плюс интерактив.

Запуск:  streamlit run app/streamlit_app.py      (из корня проекта)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import streamlit as st

# чтобы импорт app.data работал при запуске из корня проекта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.data import load_clean_data

st.set_page_config(page_title="Lending Club — анализ", page_icon="🏦", layout="wide")
sns.set_theme(style="whitegrid", palette="deep")


@st.cache_data(show_spinner="Загружаю и чищу данные…")
def get_data():
    return load_clean_data()


data = get_data()
GRADE_ORDER = sorted(data["grade"].unique())

st.sidebar.title("🏦 Lending Club")
st.sidebar.caption("Итоговый проект · Python for Data Science DSBA")
SECTION = st.sidebar.radio("Разделы отчёта", [
    "Аннотация",
    "1. Описание датасета",
    "2. Описательная статистика",
    "3. Очистка данных",
    "4. Графики числовых полей",
    "5. Детальный обзор",
    "6. Трансформация данных",
    "7. Проверка гипотез",
    "8. Выводы",
    "🔎 Интерактивный обзор",
    "➕ Создать кредит",
])


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
if SECTION == "Аннотация":
    st.title("🏦 Анализ данных Lending Club: от чего зависит дефолт по кредиту")
    st.markdown("""
**Курс:** Python for Data Science (DSBA), 2025/2026 — итоговый проект
**Датасет:** [Lending Club Loan Data](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv) (Kaggle)

> **Автор проекта:** _укажи свои ФИО и группу_.

### Аннотация
Lending Club — крупнейшая в США платформа однорангового (P2P) кредитования. Для инвестора
главный вопрос — **вернут ли деньги**. На 396 тысячах выданных кредитов мы исследуем, какие
характеристики заёмщика и займа связаны с его исходом (полное погашение или дефолт — *Charged
Off*). Проводим полный цикл: описание данных и их качества, описательная статистика, очистка,
разведочные и сравнительные графики, конструирование новых признаков и статистическая проверка
гипотез о факторах риска.

Главный сюжет — насколько **грейд** (рейтинг A–G) и **ставка** отражают реальный риск и
сохраняется ли влияние срока и ставки на дефолт *при прочих равных*. **Вклад:** проект выполнен
одним автором (предобработка, анализ, визуализация, проверка гипотез, оформление).
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("Кредитов (после очистки)", f"{len(data):,}")
    c2.metric("Доля дефолтов", f"{data['is_default'].mean()*100:.1f}%")
    c3.metric("Числовых признаков", "13+")

# ─────────────────────────────────────────────────────────────────────────
elif SECTION == "1. Описание датасета":
    st.header("1. Описание датасета")
    st.markdown("""
**Предметная область:** потребительское кредитование (финтех, P2P-займы). Каждая строка — один
выданный кредит: его параметры (сумма, ставка, срок, грейд), данные о заёмщике (доход, стаж,
жильё, долговая нагрузка) и **итог** — погашен или дефолт.
""")
    st.subheader("Первые строки")
    st.dataframe(data.head(20), width="stretch")

    st.subheader("Качество данных")
    st.markdown("""
Исходные данные требовали обработки: пропуски в нескольких колонках (`mort_acc` ~9.5%,
`emp_length`, `emp_title`, `title`, `pub_rec_bankruptcies`, `revol_util`); поля `term` и
`emp_length` хранились как текст; в `dti` была заглушка `9999`, у `annual_inc` — тяжёлый
правый хвост. Всё это поправлено на этапе очистки (см. раздел 3).
""")
    c1, c2 = st.columns(2)
    c1.metric("Строк", f"{len(data):,}")
    c2.metric("Колонок (после фич-инжиниринга)", f"{data.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────
elif SECTION == "2. Описательная статистика":
    st.header("2. Описательная статистика числовых полей")
    num_fields = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "revol_util"]
    desc = data[num_fields].describe().T
    desc["median"] = data[num_fields].median()
    desc = desc[["count", "mean", "median", "std", "min", "25%", "75%", "max"]]
    st.dataframe(desc.round(2), width="stretch")
    st.markdown("""
**Что видно.** Типичный кредит — около \\$14k (медиана \\$12k) под ~13–14% с платежом ~\\$430/мес.
Доход сильно скошен (среднее \\$74k > медианы \\$64k). У `revol_util` заёмщики используют в
среднем ~54% кредитного лимита.
""")

# ─────────────────────────────────────────────────────────────────────────
elif SECTION == "3. Очистка данных":
    st.header("3. Очистка данных")
    st.markdown("""
Шаги предобработки (реализованы в `app/data.py`, единые для ноутбука и веба):
1. `term` « 36 months» → число `36`; `emp_length` текст → число лет.
2. Из `address` извлечён штат (`state`); введена цель `is_default` (1 = Charged Off).
3. Пропуски в `mort_acc`/`revol_util` заполнены медианой, в `pub_rec_bankruptcies` — нулём.
4. Удалён свободный текст (`emp_title`, `title`, `address`) и остаточные строки с пропусками.
5. Убраны аномалии: `dti > 60` (заглушки `9999`) и нулевой доход.

Потеряно менее процента данных; пропусков не осталось, типы корректны.
""")
    st.success(f"После очистки: {len(data):,} строк × {data.shape[1]} колонок, пропусков нет.")

# ─────────────────────────────────────────────────────────────────────────
elif SECTION == "4. Графики числовых полей":
    st.header("4. Графики числовых полей (5 полей, 3 типа)")

    st.subheader("Гистограммы: сумма и доход")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(data["loan_amnt"], bins=40, kde=True, color="#337ab7", ax=axes[0])
    axes[0].set_title("Сумма кредита"); axes[0].set_xlabel("Сумма, $")
    sns.histplot(data[data["annual_inc"] <= 250_000]["annual_inc"], bins=40,
                 color="#5cb85c", ax=axes[1])
    axes[1].set_title("Годовой доход (до $250k)"); axes[1].set_xlabel("Доход, $")
    show(fig)

    st.subheader("Гистограммы: ставка и долговая нагрузка")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(data["int_rate"], bins=40, kde=True, color="#f0ad4e", ax=axes[0])
    axes[0].set_title("Ставка, %"); axes[0].set_xlabel("Ставка, %")
    sns.histplot(data["dti"], bins=40, kde=True, color="#9b59b6", ax=axes[1])
    axes[1].set_title("DTI, %"); axes[1].set_xlabel("DTI, %")
    show(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Boxplot: платёж по сроку")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(data=data, x="term_months", y="installment", palette="Blues", ax=ax)
        ax.set_xlabel("Срок, мес."); ax.set_ylabel("Платёж, $")
        show(fig)
    with col2:
        st.subheader("Scatter: сумма vs платёж")
        sample = data.sample(5000, random_state=1)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=sample, x="loan_amnt", y="installment",
                        hue="term_months", palette="coolwarm", alpha=0.5, s=18, ax=ax)
        ax.set_xlabel("Сумма, $"); ax.set_ylabel("Платёж, $")
        show(fig)
    st.markdown("Покрыто 5 числовых полей тремя типами графиков (гистограмма, boxplot, scatter).")

# ─────────────────────────────────────────────────────────────────────────
elif SECTION == "5. Детальный обзор":
    st.header("5. Детальный обзор: сравнения и связи")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Доля дефолтов по грейдам")
        dbg = (data.groupby("grade")["is_default"].mean() * 100).round(1)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(x=dbg.index, y=dbg.values, palette="rocket", ax=ax)
        for i, v in enumerate(dbg.values):
            ax.text(i, v + 0.3, f"{v}%", ha="center")
        ax.set_xlabel("Грейд"); ax.set_ylabel("Доля дефолтов, %")
        show(fig)
    with col2:
        st.subheader("Ставка по грейдам")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(data=data, x="grade", y="int_rate", order=GRADE_ORDER,
                    palette="viridis", ax=ax)
        ax.set_xlabel("Грейд"); ax.set_ylabel("Ставка, %")
        show(fig)

    st.subheader("Тепловая карта: доля дефолтов грейд × срок")
    pivot = data.pivot_table(index="grade", columns="term_months",
                             values="is_default", aggfunc="mean") * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="rocket_r", ax=ax,
                cbar_kws={"label": "Доля дефолтов, %"})
    ax.set_xlabel("Срок, мес."); ax.set_ylabel("Грейд")
    show(fig)

    st.subheader("Матрица корреляций")
    num_cols = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti",
                "open_acc", "revol_bal", "revol_util", "total_acc",
                "mort_acc", "term_months", "emp_length_num", "is_default"]
    corr = data[num_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax)
    show(fig)

    st.subheader("Сводная статистика по грейдам")
    summary = data.groupby("grade").agg(
        кол_во=("grade", "size"), ставка=("int_rate", "mean"),
        сумма=("loan_amnt", "mean"), доход=("annual_inc", "mean"),
        DTI=("dti", "mean"), доля_дефолтов=("is_default", "mean")).round(2)
    summary["доля_дефолтов"] = (summary["доля_дефолтов"] * 100).round(1)
    st.dataframe(summary, width="stretch")

    st.markdown("""
**Обсуждение.** Грейд — рабочая мера риска: дефолты растут от A (~6%) к G (~48%), и ставка
растёт так же. Длинный срок и плохой грейд вместе дают худший сценарий. Сильнее всего риск
повышают ставка и срок, снижают — ипотечные счета.
""")

# ─────────────────────────────────────────────────────────────────────────
elif SECTION == "6. Трансформация данных":
    st.header("6. Трансформация данных: новые признаки")
    st.markdown("""
Сконструированы новые колонки:
- **`grade_num`** — грейд A–G → 1–7;
- **`total_payment`** = `installment` × `term_months`;
- **`total_interest`** = `total_payment` − `loan_amnt` (переплата, $);
- **`overpay_ratio`** = `total_interest` / `loan_amnt`;
- **`payment_to_income`** = `installment` × 12 / `annual_inc`;
- **`credit_history_years`** — длина кредитной истории на момент выдачи.
""")
    st.dataframe(
        data[["grade", "grade_num", "loan_amnt", "total_payment", "total_interest",
              "overpay_ratio", "payment_to_income", "credit_history_years"]].head(15),
        width="stretch")
    st.dataframe(
        data[["overpay_ratio", "payment_to_income", "credit_history_years",
              "total_interest"]].describe().round(2).T, width="stretch")

# ─────────────────────────────────────────────────────────────────────────
elif SECTION == "7. Проверка гипотез":
    st.header("7. Проверка гипотез")
    st.caption("Уровень значимости α = 0.05, тесты — scipy.stats")

    st.subheader("Гипотеза 1. Длинный срок повышает риск дефолта при одинаковом грейде")
    rows = []
    for g in GRADE_ORDER:
        sub = data[data["grade"] == g]
        ct = pd.crosstab(sub["term_months"], sub["is_default"])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        dr36 = sub[sub["term_months"] == 36]["is_default"].mean() * 100
        dr60 = sub[sub["term_months"] == 60]["is_default"].mean() * 100
        rows.append([g, round(dr36, 1), round(dr60, 1), round(dr60 - dr36, 1), p])
    h1 = pd.DataFrame(rows, columns=["грейд", "дефолт_36_%", "дефолт_60_%", "разница_пп", "p_value"])
    st.dataframe(h1, width="stretch")
    plot_df = data.groupby(["grade", "term_months"])["is_default"].mean().mul(100).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=plot_df, x="grade", y="is_default", hue="term_months", palette="Set1", ax=ax)
    ax.set_xlabel("Грейд"); ax.set_ylabel("Доля дефолтов, %"); ax.legend(title="Срок, мес.")
    show(fig)
    st.success("Вывод H1: подтвердилась. В грейдах A–F 60 мес. дефолтит значимо чаще (p<0.05); "
               "исключение — малочисленный G (p≈0.83). Срок — самостоятельный фактор риска.")

    st.subheader("Гипотеза 2. Ставка несёт информацию о риске сверх грейда")
    rows = []
    for g in GRADE_ORDER:
        sub = data[data["grade"] == g]
        r, p = stats.pointbiserialr(sub["is_default"], sub["int_rate"])
        med = sub["int_rate"].median()
        low = sub[sub["int_rate"] <= med]["is_default"].mean() * 100
        high = sub[sub["int_rate"] > med]["is_default"].mean() * 100
        rows.append([g, round(r, 3), p, round(low, 1), round(high, 1)])
    h2 = pd.DataFrame(rows, columns=["грейд", "корр_r", "p_value", "дефолт_дешёвые_%", "дефолт_дорогие_%"])
    st.dataframe(h2, width="stretch")
    st.warning("Вывод H2: в чистом виде НЕ подтвердилась. Внутри грейда |r| ≤ 0.07, знак "
               "неоднозначен (в D и E отрицательный). После фиксации грейда ставка почти не "
               "несёт доп. информации о риске — нормальный результат.")

    st.subheader("Гипотеза 3 (бонус). Платёжная нагрузка к доходу связана с дефолтом")
    r, p = stats.pointbiserialr(data["is_default"], data["payment_to_income"])
    st.write(f"Корреляция payment_to_income ↔ is_default: **r = {r:.3f}**, p = {p:.2e}")
    q = pd.qcut(data["payment_to_income"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    dr = (data.groupby(q)["is_default"].mean() * 100).round(1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=dr.index, y=dr.values, palette="rocket", ax=ax)
    for i, v in enumerate(dr.values):
        ax.text(i, v + 0.2, f"{v}%", ha="center")
    ax.set_xlabel("Квартиль платёж/доход"); ax.set_ylabel("Доля дефолтов, %")
    show(fig)
    st.success("Вывод H3: подтвердилась. Доля дефолтов растёт от Q1 к Q4 (r>0, p≪0.05).")

# ─────────────────────────────────────────────────────────────────────────
elif SECTION == "8. Выводы":
    st.header("8. Выводы")
    st.markdown("""
1. **Грейд — рабочая мера риска.** Дефолты растут от A (~6%) к G (~48%), ставка — так же.
2. **Срок — самостоятельный фактор риска** (H1): 60 мес. хуже 36 *внутри каждого грейда*
   (значимо в A–F).
3. **Ставка почти не добавляет информации сверх грейда** (H2): связь близка к нулю.
4. **Платёжная нагрузка к доходу важна** (H3): больше доля платежа — выше риск.
5. **Слабо влияют:** стаж работы и формальная проверка дохода.

**Практический смысл.** Ключевые сигналы риска — грейд/ставка, срок и платёжная нагрузка.
""")

# ─────────────────────────────────────────────────────────────────────────
elif SECTION == "🔎 Интерактивный обзор":
    st.header("🔎 Интерактивный обзор данных")
    st.caption("Фильтруй выборку — статистика и график пересчитываются на лету.")
    c1, c2, c3 = st.columns(3)
    grades = c1.multiselect("Грейды", GRADE_ORDER, default=GRADE_ORDER)
    terms = c2.multiselect("Срок, мес.", [36, 60], default=[36, 60])
    purposes = c3.multiselect("Цель", sorted(data["purpose"].unique()),
                              default=list(data["purpose"].value_counts().head(3).index))
    amnt = st.slider("Диапазон суммы кредита, $", 0, int(data["loan_amnt"].max()),
                     (0, int(data["loan_amnt"].max())), step=1000)

    f = data[data["grade"].isin(grades) & data["term_months"].isin(terms)
             & data["purpose"].isin(purposes)
             & data["loan_amnt"].between(amnt[0], amnt[1])]
    if len(f) == 0:
        st.warning("Под фильтр ничего не попало.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Кредитов", f"{len(f):,}")
        m2.metric("Доля дефолтов", f"{f['is_default'].mean()*100:.1f}%")
        m3.metric("Средняя ставка", f"{f['int_rate'].mean():.1f}%")
        m4.metric("Средняя сумма", f"${f['loan_amnt'].mean():,.0f}")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(f["int_rate"], bins=40, kde=True, color="#337ab7", ax=ax)
        ax.set_title("Распределение ставки в выборке"); ax.set_xlabel("Ставка, %")
        show(fig)
        st.dataframe(f.head(50), width="stretch")

# ─────────────────────────────────────────────────────────────────────────
elif SECTION == "➕ Создать кредит":
    st.header("➕ Создать новую запись кредита")
    st.caption("Аналог POST /loans: считаем аннуитетный платёж и переплату.")
    with st.form("new_loan"):
        c1, c2, c3 = st.columns(3)
        loan_amnt = c1.number_input("Сумма, $", 500.0, 40000.0, 12000.0, step=500.0)
        int_rate = c2.number_input("Ставка, %", 1.0, 35.0, 13.5, step=0.5)
        term = c3.selectbox("Срок, мес.", [36, 60])
        c4, c5, c6 = st.columns(3)
        annual_inc = c4.number_input("Годовой доход, $", 1000.0, 1_000_000.0, 65000.0, step=1000.0)
        grade = c5.selectbox("Грейд", GRADE_ORDER)
        purpose = c6.selectbox("Цель", sorted(data["purpose"].unique()))
        submitted = st.form_submit_button("Рассчитать и создать")
    if submitted:
        r = int_rate / 100 / 12
        installment = loan_amnt * r * (1 + r) ** term / ((1 + r) ** term - 1) if r > 0 else loan_amnt / term
        total_payment = installment * term
        overpay = (total_payment - loan_amnt) / loan_amnt
        pti = installment * 12 / annual_inc
        c1, c2, c3 = st.columns(3)
        c1.metric("Ежемесячный платёж", f"${installment:,.2f}")
        c2.metric("Переплата", f"{overpay*100:.1f}%")
        c3.metric("Платёж / доход (год)", f"{pti*100:.1f}%")
        st.success("Запись создана (демонстрация формы-обработчика).")
        st.json({"loan_amnt": loan_amnt, "term_months": term, "int_rate": int_rate,
                 "grade": grade, "annual_inc": annual_inc, "purpose": purpose,
                 "installment": round(installment, 2), "overpay_ratio": round(overpay, 3)})
