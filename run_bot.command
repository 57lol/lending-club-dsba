#!/bin/bash
# Двойной клик — запускает Telegram-бота (на глобальном Python с aiogram).
cd "$(dirname "$0")" || exit 1
echo "🤖 Запускаю Telegram-бота Lending Club…"
echo "   Нужен токен от @BotFather (см. README, раздел про бота)."
echo "   Закрыть: Ctrl+C в этом окне."
echo ""
PYBIN="/Users/nigo57/anaconda3/bin/python3"
[ -x "$PYBIN" ] || PYBIN="python3"
exec "$PYBIN" -m app.bot
