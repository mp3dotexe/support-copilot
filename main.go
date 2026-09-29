package main

import (
	_ "embed"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"os"
	"strings"
)

//go:embed index.html
var indexHTML []byte

const maxBodyBytes = 64 << 10

type handleRequest struct {
	History []Message `json:"history"`
	Message string    `json:"message"`
}

func main() {
	loadDotEnv(".env")
	kb, err := loadKB("knowledge_base.json")
	if err != nil {
		log.Fatalf("load knowledge base: %v", err)
	}

	llm := newLLM(getenv("LLM_BASE_URL", "http://localhost:11434/v1"), os.Getenv("LLM_API_KEY"), getenv("MODEL", "qwen2.5:7b"), kb)
	if err := llm.probe(); err != nil {
		llm.mock = true
		log.Printf("mode: MOCK (LLM endpoint %s unreachable: %v)", llm.baseURL, err)
	} else {
		log.Printf("mode: LIVE (endpoint %s, model %s)", llm.baseURL, llm.model)
	}

	http.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(indexHTML)
	})
	// Mode/model for the header badge and KB topic names for the source chips.
	http.HandleFunc("GET /api/info", func(w http.ResponseWriter, r *http.Request) {
		topics := map[string]string{}
		for _, e := range kb {
			topics[e.ID] = e.Topic
		}
		writeJSON(w, http.StatusOK, map[string]any{"mode": llm.mode(), "model": llm.model, "topics": topics})
	})
	http.HandleFunc("POST /api/handle", handleAPI(llm))
	http.HandleFunc("POST /api/amocrm/webhook", handleAmoWebhook(llm))

	log.Printf("listening on http://localhost:8080 (%d KB entries)", len(kb))
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func handleAPI(llm *LLM) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Copilot-Mode", llm.mode())
		var req handleRequest
		if !decodeJSON(w, r, &req) {
			return
		}
		if strings.TrimSpace(req.Message) == "" {
			writeError(w, http.StatusBadRequest, "message is required")
			return
		}
		res, err := llm.Handle(r.Context(), req.History, req.Message)
		if err != nil {
			writeError(w, http.StatusBadGateway, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, res)
	}
}

// decodeJSON reads a size-limited JSON body and writes a 400/413 on failure.
func decodeJSON(w http.ResponseWriter, r *http.Request, v any) bool {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
	err := json.NewDecoder(r.Body).Decode(v)
	if err == nil {
		return true
	}
	var tooBig *http.MaxBytesError
	if errors.As(err, &tooBig) {
		writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
	} else {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
	}
	return false
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

// loadDotEnv sets KEY=VALUE pairs from a .env file if it exists; real env vars win.
func loadDotEnv(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		k = strings.TrimSpace(k)
		if _, set := os.LookupEnv(k); !set {
			os.Setenv(k, strings.Trim(strings.TrimSpace(v), `"'`))
		}
	}
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
