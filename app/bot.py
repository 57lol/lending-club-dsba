# -*- coding: utf-8 -*-
"""
Telegram-бот к проекту Lending Club (aiogram 3.x).

Меню с несколькими страницами — можно получить любую часть проекта:
описание, описательную статистику, графики (картинками), статистику по грейдам,
проверку гипотез, выводы, а также создать новую запись кредита (форма-обработчик).

Токен бота:
  • переменная окружения  LENDING_BOT_TOKEN, либо
  • файл  app/bot_token.txt  (одна строка с токеном от @BotFather).

Запуск:  python -m app.bot           (из корня проекта, на глобальном Python с aiogram)
"""
import os
import sys
import io
import asyncio
import logging

import matplotlib
matplotlib.use("Agg")                       # без GUI — рисуем в память
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (Message, CallbackQuery, BufferedInputFile,
                           InlineKeyboardButton, InlineKeyboardMarkup)
from aiogram.utils.keyboard import InlineKeyboardBuilder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.data import load_clean_data

logging.basicConfig(level=logging.INFO)
sns.set_theme(style="whitegrid", palette="deep")

DATA = load_clean_data()
GRADE_ORDER = sorted(DATA["grade"].unique())

dp = Dispatcher(storage=MemoryStorage())


# ─────────────────────────── вспомогательное ───────────────────────────
def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def money(n, dec=0) -> str:
    """Число с пробелом как разделителем тысяч (не трогая остальной текст)."""
    return f"{n:,.{dec}f}".replace(",", " ")


def menu_kb() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="📋 Описание датасета", callback_data="about")
    kb.button(text="📊 Описательная статистика", callback_data="stats")
    kb.button(text="📈 Графики", callback_data="plots")
    kb.button(text="🏆 Дефолты по грейдам", callback_data="grades")
    kb.button(text="🔬 Проверка гипотез", callback_data="hypo")
    kb.button(text="📝 Выводы", callback_data="conclusions")
    kb.button(text="➕ Создать кредит", callback_data="create")
    kb.adjust(1)
    return kb.as_markup()


def back_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⬅️ В меню", callback_data="menu")]])


WELCOME = (
    "🏦 <b>Lending Club — анализ кредитов</b>\n\n"
    "Итоговый проект по курсу Python for Data Science (DSBA).\n"
    f"В базе <b>{money(len(DATA))}</b> кредитов, доля дефолтов — "
    f"<b>{DATA['is_default'].mean()*100:.1f}%</b>.\n\n"
    "Выбери раздел отчёта:"
)


# ─────────────────────────── команды и меню ───────────────────────────
@dp.message(Command("start", "menu", "help"))
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(WELCOME, reply_markup=menu_kb())


@dp.callback_query(F.data == "menu")
async def cb_menu(cb: CallbackQuery, state: FSMContext):
    await state.clear()
    await cb.message.answer(WELCOME, reply_markup=menu_kb())
    await cb.answer()


@dp.callback_query(F.data == "about")
async def cb_about(cb: CallbackQuery):
    txt = (
        "📋 <b>Описание датасета</b>\n\n"
        "Предметная область — потребительское P2P-кредитование (Lending Club, США). "
        "Каждая строка = один выданный кредит: сумма, ставка, срок, грейд A–G, "
        "доход и долговая нагрузка заёмщика, цель кредита и <b>итог</b> "
        "(погашен / дефолт «Charged Off»).\n\n"
        f"После очистки: <b>{money(len(DATA))}</b> строк, <b>{DATA.shape[1]}</b> колонок, "
        "пропусков нет. Числовых полей 13+."
    )
    await cb.message.answer(txt, reply_markup=back_kb())
    await cb.answer()


@dp.callback_query(F.data == "stats")
async def cb_stats(cb: CallbackQuery):
    fields = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "revol_util"]
    lines = ["📊 <b>Описательная статистика</b>\n", "<pre>поле          mean    median    std</pre>"]
    for c in fields:
        s = DATA[c]
        lines.append(f"<pre>{c:12} {s.mean():8.1f} {s.median():8.1f} {s.std():8.1f}</pre>")
    await cb.message.answer("\n".join(lines), reply_markup=back_kb())
    await cb.answer()


@dp.callback_query(F.data == "plots")
async def cb_plots(cb: CallbackQuery):
    await cb.answer("Строю графики…")

    # 1) доля дефолтов по грейдам
    dbg = (DATA.groupby("grade")["is_default"].mean() * 100).round(1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=dbg.index, y=dbg.values, palette="rocket", ax=ax)
    for i, v in enumerate(dbg.values):
        ax.text(i, v + 0.3, f"{v}%", ha="center")
    ax.set_title("Доля дефолтов по грейдам"); ax.set_xlabel("Грейд"); ax.set_ylabel("%")
    await cb.message.answer_photo(
        BufferedInputFile(fig_to_png(fig), "p1.png"),
        caption="Грейд = мера риска: от A (~6%) к G (~48%).")

    # 2) ставка по грейдам
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=DATA, x="grade", y="int_rate", order=GRADE_ORDER, palette="viridis", ax=ax)
    ax.set_title("Ставка по грейдам"); ax.set_xlabel("Грейд"); ax.set_ylabel("Ставка, %")
    await cb.message.answer_photo(
        BufferedInputFile(fig_to_png(fig), "p2.png"),
        caption="Чем хуже грейд — тем выше ставка: риск заложен в цену.")

    # 3) тепловая карта грейд × срок
    pivot = DATA.pivot_table(index="grade", columns="term_months",
                             values="is_default", aggfunc="mean") * 100
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="rocket_r", ax=ax,
                cbar_kws={"label": "Доля дефолтов, %"})
    ax.set_title("Дефолты: грейд × срок"); ax.set_xlabel("Срок, мес."); ax.set_ylabel("Грейд")
    await cb.message.answer_photo(
        BufferedInputFile(fig_to_png(fig), "p3.png"),
        caption="Худший сценарий — плохой грейд и длинный срок одновременно.",
        reply_markup=back_kb())


@dp.callback_query(F.data == "grades")
async def cb_grades(cb: CallbackQuery):
    g = DATA.groupby("grade").agg(
        кол=("grade", "size"), ставка=("int_rate", "mean"),
        дефолт=("is_default", "mean"))
    lines = ["🏆 <b>Статистика по грейдам</b>\n",
             "<pre>гр   кол-во  ставка  дефолт</pre>"]
    for gr, row in g.iterrows():
        lines.append(f"<pre>{gr}  {int(row['кол']):7d}  {row['ставка']:5.1f}%  "
                     f"{row['дефолт']*100:5.1f}%</pre>")
    await cb.message.answer("\n".join(lines), reply_markup=back_kb())
    await cb.answer()


@dp.callback_query(F.data == "hypo")
async def cb_hypo(cb: CallbackQuery):
    await cb.answer("Считаю тесты…")
    # H1 — срок внутри грейда
    sig = 0
    for gr in GRADE_ORDER:
        sub = DATA[DATA["grade"] == gr]
        ct = pd.crosstab(sub["term_months"], sub["is_default"])
        _, p, _, _ = stats.chi2_contingency(ct)
        if p < 0.05:
            sig += 1
    # H3 — нагрузка/доход
    r3, p3 = stats.pointbiserialr(DATA["is_default"], DATA["payment_to_income"])
    txt = (
        "🔬 <b>Проверка гипотез</b> (α = 0.05)\n\n"
        "<b>H1.</b> Длинный срок (60 мес.) повышает риск дефолта <i>при одинаковом грейде</i>.\n"
        f"→ Значимо в {sig} из {len(GRADE_ORDER)} грейдов (A–F). "
        "<b>Подтвердилась.</b>\n\n"
        "<b>H2.</b> Ставка несёт риск сверх грейда.\n"
        "→ Внутри грейда |r| ≤ 0.07, знак неоднозначен. "
        "<b>В чистом виде не подтвердилась</b> — грейд уже вобрал риск.\n\n"
        "<b>H3.</b> Платёжная нагрузка к доходу связана с дефолтом.\n"
        f"→ r = {r3:.3f}, p = {p3:.1e}. <b>Подтвердилась.</b>"
    )
    await cb.message.answer(txt, reply_markup=back_kb())


@dp.callback_query(F.data == "conclusions")
async def cb_conclusions(cb: CallbackQuery):
    txt = (
        "📝 <b>Выводы</b>\n\n"
        "1. Грейд — рабочая мера риска (дефолты A ~6% → G ~48%).\n"
        "2. Срок — самостоятельный фактор риска (H1): 60 мес. хуже 36 внутри каждого грейда.\n"
        "3. Ставка почти не добавляет информации сверх грейда (H2).\n"
        "4. Платёжная нагрузка к доходу важна (H3).\n"
        "5. Слабо влияют стаж работы и формальная проверка дохода.\n\n"
        "<i>Ключевые сигналы риска: грейд/ставка, срок и платёжная нагрузка.</i>"
    )
    await cb.message.answer(txt, reply_markup=back_kb())
    await cb.answer()


# ─────────────────────────── форма: создать кредит ───────────────────────────
class NewLoan(StatesGroup):
    amount = State()
    rate = State()
    term = State()
    income = State()


@dp.callback_query(F.data == "create")
async def cb_create(cb: CallbackQuery, state: FSMContext):
    await state.set_state(NewLoan.amount)
    await cb.message.answer("➕ <b>Новый кредит</b>\n\nВведи сумму кредита, $ (например, 12000):")
    await cb.answer()


async def _ask_float(message: Message, state: FSMContext, lo, hi, nxt, prompt):
    try:
        val = float(message.text.replace(",", ".").replace(" ", ""))
    except ValueError:
        await message.answer("Нужно число. Попробуй ещё раз:")
        return None
    if not (lo <= val <= hi):
        await message.answer(f"Значение должно быть в диапазоне {lo}–{hi}. Ещё раз:")
        return None
    return val


@dp.message(NewLoan.amount)
async def loan_amount(message: Message, state: FSMContext):
    v = await _ask_float(message, state, 500, 40000, None, None)
    if v is None:
        return
    await state.update_data(amount=v)
    await state.set_state(NewLoan.rate)
    await message.answer("Введи годовую ставку, % (например, 13.5):")


@dp.message(NewLoan.rate)
async def loan_rate(message: Message, state: FSMContext):
    v = await _ask_float(message, state, 1, 35, None, None)
    if v is None:
        return
    await state.update_data(rate=v)
    await state.set_state(NewLoan.term)
    await message.answer("Введи срок: <b>36</b> или <b>60</b> месяцев:")


@dp.message(NewLoan.term)
async def loan_term(message: Message, state: FSMContext):
    t = message.text.strip()
    if t not in ("36", "60"):
        await message.answer("Только 36 или 60. Ещё раз:")
        return
    await state.update_data(term=int(t))
    await state.set_state(NewLoan.income)
    await message.answer("Введи годовой доход, $ (например, 65000):")


@dp.message(NewLoan.income)
async def loan_income(message: Message, state: FSMContext):
    v = await _ask_float(message, state, 1000, 5_000_000, None, None)
    if v is None:
        return
    d = await state.get_data()
    await state.clear()

    amount, rate, term, income = d["amount"], d["rate"], d["term"], v
    r = rate / 100 / 12
    installment = amount * r * (1 + r) ** term / ((1 + r) ** term - 1) if r > 0 else amount / term
    total = installment * term
    overpay = (total - amount) / amount
    pti = installment * 12 / income

    txt = (
        "✅ <b>Кредит создан</b>\n\n"
        f"Сумма: <b>${money(amount)}</b>\n"
        f"Ставка: <b>{rate}%</b> · Срок: <b>{term} мес.</b>\n"
        f"Доход: <b>${money(income)}</b>\n\n"
        f"💳 Ежемесячный платёж: <b>${money(installment, 2)}</b>\n"
        f"💰 Переплата: <b>{overpay*100:.1f}%</b> (${money(total-amount)})\n"
        f"📉 Платёж/доход (год): <b>{pti*100:.1f}%</b>"
    )
    await message.answer(txt, reply_markup=back_kb())


# ─────────────────────────── запуск ───────────────────────────
def get_token() -> str:
    token = os.getenv("LENDING_BOT_TOKEN", "").strip()
    if not token:
        path = os.path.join(os.path.dirname(__file__), "bot_token.txt")
        if os.path.exists(path):
            token = open(path, encoding="utf-8").read().strip()
    return token


async def main():
    token = get_token()
    if not token:
        print("\n[!] Не найден токен бота.\n"
              "    1) Получи токен у @BotFather в Telegram.\n"
              "    2) Либо экспортируй:  export LENDING_BOT_TOKEN=ТОКЕН\n"
              "       либо создай файл  app/bot_token.txt  с токеном внутри.\n"
              "    3) Запусти снова:  python -m app.bot\n")
        sys.exit(1)
    bot = Bot(token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    print("Бот запущен. Открой его в Telegram и нажми /start. Ctrl+C — остановить.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
