#!/bin/bash
# Двойной клик — запускает веб-отчёт (Streamlit) в браузере.
cd "$(dirname "$0")" || exit 1
echo "🏦 Запускаю веб-отчёт Lending Club (Streamlit)…"
echo "   Откроется в браузере: http://localhost:8501"
echo "   Закрыть: Ctrl+C в этом окне."
echo ""
exec ./.venv/bin/python -m streamlit run app/streamlit_app.py
