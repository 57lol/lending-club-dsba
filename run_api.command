#!/bin/bash
# Двойной клик — запускает REST API (FastAPI).
cd "$(dirname "$0")" || exit 1
echo "🔌 Запускаю REST API (FastAPI)…"
echo "   Документация Swagger: http://localhost:8000/docs"
echo "   Примеры: http://localhost:8000/loans?grade=B&limit=5"
echo "   Закрыть: Ctrl+C в этом окне."
echo ""
exec ./.venv/bin/python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
