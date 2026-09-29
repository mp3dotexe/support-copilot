package main

import (
	"fmt"
	"net/http"
	"strings"
	"time"
)

// AmoCRM integration MOCK: nothing is sent to AmoCRM. The handler returns the note
// that would be posted back. A real integration would receive chat messages via an
// AmoCRM widget or webhook and write the note with POST /api/v4/leads/notes (OAuth token).

// amoIncoming is a simplified AmoCRM-style incoming chat message.
type amoIncoming struct {
	LeadID  int64 `json:"lead_id"`
	Contact struct {
		Name string `json:"name"`
	} `json:"contact"`
	Message struct {
		Text      string `json:"text"`
		CreatedAt int64  `json:"created_at"` // unix seconds
	} `json:"message"`
}

// amoNote mirrors one item of the AmoCRM v4 notes API payload.
type amoNote struct {
	EntityID int64  `json:"entity_id"`
	NoteType string `json:"note_type"`
	Params   struct {
		Text string `json:"text"`
	} `json:"params"`
	CreatedAt int64 `json:"created_at"`
}

// Russian display labels for the note text; API values stay English (same mapping as index.html).
var (
	intentLabels = map[string]string{"price_question": "Вопрос о цене", "delivery": "Доставка", "complaint": "Жалоба", "order": "Заказ", "other": "Другое"}
	stageLabels  = map[string]string{"browsing": "Присматривается", "considering": "Выбирает", "ready_to_buy": "Готов купить", "post_sale": "После продажи"}
)

func handleAmoWebhook(llm *LLM) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Copilot-Mode", llm.mode())
		var in amoIncoming
		if !decodeJSON(w, r, &in) {
			return
		}
		if in.LeadID == 0 || strings.TrimSpace(in.Message.Text) == "" {
			writeError(w, http.StatusBadRequest, "lead_id and message.text are required")
			return
		}
		res, err := llm.Handle(r.Context(), nil, in.Message.Text)
		if err != nil {
			writeError(w, http.StatusBadGateway, err.Error())
			return
		}

		var b strings.Builder
		fmt.Fprintf(&b, "Copilot · %s\n\nЧерновик ответа:\n%s\n\nПодсказка менеджеру:\n%s\n\n", in.Contact.Name, res.Reply, res.UpsellHint)
		fmt.Fprintf(&b, "Намерение: %s | Этап: %s | Уверенность: %.2f | Источники: %s",
			intentLabels[res.Intent], stageLabels[res.Stage], res.Confidence, strings.Join(res.Sources, ", "))
		if res.NeedsManager {
			b.WriteString("\nНет в базе знаний, нужен менеджер")
		}

		note := amoNote{EntityID: in.LeadID, NoteType: "common", CreatedAt: time.Now().Unix()}
		note.Params.Text = b.String()
		writeJSON(w, http.StatusOK, note)
	}
}
