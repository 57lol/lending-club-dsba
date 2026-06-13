# 🏦 Lending Club — анализ кредитов

Итоговый проект по курсу **Python for Data Science (DSBA), 2025/2026**.
Датасет: [Lending Club Loan Data](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv) (Kaggle).

Исследуем, какие характеристики заёмщика и кредита связаны с дефолтом (исход *Charged Off*).
В работе — полный цикл: описание и очистка данных, описательная статистика, разведочные и
сравнительные графики, конструирование признаков и статистическая проверка гипотез.

> ⚠️ **Перед сдачей:** открой ноутбук и впиши свои **ФИО и группу** в самой первой ячейке
> (раздел «Аннотация»). То же — в Streamlit (раздел «Аннотация») и при желании в боте.

---

## Что внутри

```
Lending_Club_Project/
├── Lending_Club_Project.ipynb   ← ГЛАВНЫЙ отчётный ноутбук (выполнен, с графиками)
├── build_notebook.py            ← генератор ноутбука (для воспроизводимости)
├── data/
│   └── lending_club_loan.csv    ← датасет (396 030 строк × 27 колонок)
├── app/
│   ├── data.py                  ← загрузка + очистка + новые признаки (общий модуль)
│   ├── streamlit_app.py         ← веб-версия отчёта (эквивалент ноутбука)
│   ├── api.py                   ← REST API (FastAPI): GET с фильтрами + POST
│   └── bot.py                   ← Telegram-бот с меню по разделам проекта
├── run_web.command              ← запуск Streamlit (двойной клик)
├── run_api.command              ← запуск REST API (двойной клик)
├── run_bot.command              ← запуск Telegram-бота (двойной клик)
├── requirements.txt             ← зависимости веб-части (читает и Streamlit Cloud)
└── requirements-notebook.txt    ← доп. зависимости ноутбука и бота (локально)
```

Соответствие требованиям курса (всё в `Lending_Club_Project.ipynb`):
аннотация с вкладом · описание датасета и качество данных · описательная статистика
(mean/median/std по 6 числовым полям) · очистка · 5 числовых полей в 3 типах графиков ·
8+ сравнительных выводов · трансформация (6 новых колонок) · 3 проверенные гипотезы
(scipy) · обсуждение на каждом шаге.

---

## 1. Главное — ноутбук

Уже выполнен, все графики встроены. Просто открой:

```bash
cd ~/Lending_Club_Project
jupyter lab Lending_Club_Project.ipynb      # или jupyter notebook
```

Пересобрать с нуля (если нужно переисполнить всё):

```bash
python3 build_notebook.py
```

---

## 2. Веб-интерфейс (Streamlit + REST API)

Веб-часть живёт в отдельном виртуальном окружении `.venv` (оно уже создано). Причина — конфликт
версий: `streamlit` требует `anyio>=4`, а классический `jupyter-server` — `anyio<4`, поэтому их
разводим по разным окружениям.

**Запуск (просто двойной клик в Finder):**
- `run_web.command` → веб-отчёт на http://localhost:8501
- `run_api.command` → REST API + Swagger на http://localhost:8000/docs

**Или из терминала:**

```bash
cd ~/Lending_Club_Project
./.venv/bin/python -m streamlit run app/streamlit_app.py      # веб-отчёт
./.venv/bin/python -m uvicorn app.api:app --port 8000         # REST API
```

Если `.venv` вдруг нет — создать заново:

```bash
python3 -m venv --system-site-packages .venv
./.venv/bin/python -m pip install -r requirements.txt
```

### REST API — примеры

```bash
# GET с фильтрами и пагинацией (несколько аргументов)
curl "http://localhost:8000/loans?grade=C&term=60&max_int_rate=20&limit=5&offset=0"

# GET агрегированной статистики
curl "http://localhost:8000/stats?group_by=grade&metric=default_rate"

# POST — создать новую запись кредита (считает платёж и переплату)
curl -X POST "http://localhost:8000/loans" -H "Content-Type: application/json" \
     -d '{"loan_amnt":15000,"term_months":36,"int_rate":12.5,"grade":"B","annual_inc":70000}'
```

---

## 3. Telegram-бот

Меню с несколькими страницами: описание датасета, описательная статистика, графики (картинками),
статистика по грейдам, проверка гипотез, выводы и **форма создания кредита**.

**Шаг 1 — получить токен:** напиши [@BotFather](https://t.me/BotFather) в Telegram → `/newbot` →
придумай имя → он пришлёт **токен** вида `1234567890:AaBbCc...`.

**Шаг 2 — отдать токен боту** (любой способ):
```bash
# вариант А: файл (рекомендую)
echo "СЮДА_ТОКЕН" > ~/Lending_Club_Project/app/bot_token.txt
# вариант Б: переменная окружения
export LENDING_BOT_TOKEN="СЮДА_ТОКЕН"
```

**Шаг 3 — запустить:** двойной клик по `run_bot.command` (или `python3 -m app.bot` из корня).
Затем открой бота в Telegram и нажми **/start**.

Бот работает на глобальном Python (там стоит `aiogram`), данные и графики берёт из того же
`app/data.py`, что и ноутбук со Streamlit.

---

## Публикация Streamlit в интернет (Streamlit Community Cloud)

Проект залит на GitHub. Чтобы получить публичную ссылку:
1. Зайди на **https://share.streamlit.io** → **Sign in with GitHub** (тот же аккаунт).
2. **Create app** → **Deploy a public app from GitHub**.
3. Repository: твой репозиторий · Branch: `main` · **Main file path: `app/streamlit_app.py`**.
4. **Deploy**. Через пару минут получишь ссылку вида `https://<имя>.streamlit.app` — её и сдавай.

Streamlit Cloud сам ставит зависимости из `requirements.txt`. Датасет лежит в репозитории,
загрузка оптимизирована под бесплатный тариф.

## Сдача проекта

Согласно требованиям курса отправляешь через форму:
1. ссылку на файл ноутбука `Lending_Club_Project.ipynb`;
2. файл датасета `data/lending_club_loan.csv`;
3. (для «пилотов» / ради бонуса) ссылку на онлайн-страницу — подними Streamlit и при необходимости
   опубликуй (например, через [Streamlit Community Cloud](https://streamlit.io/cloud), указав
   `app/streamlit_app.py` как точку входа).

Дедлайн: **14 июня 23:59**. Защита: **25 и 27 июня**.
