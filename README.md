# Support Copilot

Checks a customer message against `knowledge_base.json` (Russian, ₽) and returns a KB-grounded reply draft plus an upsell hint for the manager.

- **Run:** `go run .` then open http://localhost:8080 (Go 1.22+, no dependencies). Optional: `cp .env.example .env`.
- **LLM:** any OpenAI-compatible endpoint. Default is local Ollama: `ollama pull qwen2.5:7b && ollama serve`.
- **Env vars:** `LLM_BASE_URL` (default `http://localhost:11434/v1`), `LLM_API_KEY` (optional Bearer token), `MODEL` (default `qwen2.5:7b`).
- **Language:** the UI and manager hint are in Russian. The customer reply follows the customer's language (Russian by default).
- **MOCK mode:** if the endpoint is unreachable at startup, the server answers with keyword-based canned Russian responses so the demo never breaks. The mode shows in the startup log and the UI header.
- **`POST /api/handle`:** `{"history":[{"role":"customer|manager","text":"..."}],"message":"..."}` → `reply, upsell_hint, intent, stage, confidence, sources, needs_manager`.
- **`GET /api/info`:** mode, model and KB topic names (used by the UI).
- **`POST /api/amocrm/webhook`:** `{"lead_id":1,"contact":{"id":7,"name":"Анна"},"message":{"text":"...","created_at":1759140000}}` → an AmoCRM `common` note for the lead card.
- **Mocked:** AmoCRM is not called. A real integration would use an AmoCRM widget or webhook plus the notes API (`POST /api/v4/leads/notes`).
- **Eval:** with the server running, `go run ./cmd/eval` runs `testcases.json` (deterministic checks) and prints a pass/fail table and score.
