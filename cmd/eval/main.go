// Command eval runs testcases.json against a running server (go run .) and prints a pass/fail table.
// Run it from the repo root: go run ./cmd/eval
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"regexp"
	"slices"
	"strings"
	"text/tabwriter"
	"time"
	"unicode"
)

type testCase struct {
	Name                  string              `json:"name"`
	Message               string              `json:"message"`
	History               []map[string]string `json:"history,omitempty"`
	ExpectSourcesContains []string            `json:"expect_sources_contains"`
	ExpectNeedsManager    *bool               `json:"expect_needs_manager"`
	ExpectIntent          string              `json:"expect_intent"`
	ReplyMustNotContain   []string            `json:"reply_must_not_contain"`
	MustNotContain        []string            `json:"must_not_contain"` // checked in reply and hint
	HintMustContain       []string            `json:"hint_must_contain"`
	NoNewNumbers          bool                `json:"no_new_numbers"` // checked in reply and hint
	ReplyLanguage         string              `json:"reply_language"` // "ru" or "en"
}

type result struct {
	Reply        string   `json:"reply"`
	UpsellHint   string   `json:"upsell_hint"`
	Intent       string   `json:"intent"`
	Sources      []string `json:"sources"`
	NeedsManager bool     `json:"needs_manager"`
	Error        string   `json:"error"`
}

// numberRe matches numbers with optional thousands separators (space, NBSP, narrow NBSP, comma): "15 000", "1,000".
var numberRe = regexp.MustCompile(`\d+(?:[ \x{00A0}\x{202F},]\d{3})*(?:\.\d+)?`)

func main() {
	url := os.Getenv("EVAL_URL")
	if url == "" {
		url = "http://localhost:8080"
	}
	var cases []testCase
	if err := readJSON("testcases.json", &cases); err != nil {
		fail("read testcases.json: %v", err)
	}
	kbRaw, err := os.ReadFile("knowledge_base.json")
	if err != nil {
		fail("read knowledge_base.json: %v", err)
	}
	kbNumbers := numbers(string(kbRaw))

	client := &http.Client{Timeout: 150 * time.Second} // the server may retry a 60s LLM call once
	mode := ""
	passed := 0
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "CASE\tRESULT\tDETAILS")
	for _, tc := range cases {
		res, caseMode, err := call(client, url, tc)
		if err != nil {
			fail("cannot reach server at %s (start it with `go run .`): %v", url, err)
		}
		mode = caseMode
		problems := check(tc, res, kbNumbers)
		status := "PASS"
		if len(problems) > 0 {
			status = "FAIL"
		} else {
			passed++
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\n", tc.Name, status, strings.Join(problems, "; "))
	}
	tw.Flush()

	fmt.Printf("\nScore: %d/%d\n", passed, len(cases))
	if mode == "mock" {
		fmt.Println("MOCK mode, results not meaningful (the LLM endpoint was unreachable when the server started).")
	}
}

func call(client *http.Client, url string, tc testCase) (result, string, error) {
	body, _ := json.Marshal(map[string]any{"history": tc.History, "message": tc.Message})
	resp, err := client.Post(url+"/api/handle", "application/json", bytes.NewReader(body))
	if err != nil {
		return result{}, "", err
	}
	defer resp.Body.Close()
	var res result
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		res.Error = "bad response: " + err.Error()
	} else if resp.StatusCode != http.StatusOK && res.Error == "" {
		res.Error = resp.Status
	}
	return res, resp.Header.Get("X-Copilot-Mode"), nil
}

func check(tc testCase, r result, kbNumbers []string) []string {
	if r.Error != "" {
		return []string{"server error: " + r.Error}
	}
	var p []string
	for _, id := range tc.ExpectSourcesContains {
		if !slices.Contains(r.Sources, id) {
			p = append(p, fmt.Sprintf("sources %v missing %q", r.Sources, id))
		}
	}
	if tc.ExpectNeedsManager != nil && r.NeedsManager != *tc.ExpectNeedsManager {
		p = append(p, fmt.Sprintf("needs_manager=%v, want %v", r.NeedsManager, *tc.ExpectNeedsManager))
	}
	if tc.ExpectIntent != "" && r.Intent != tc.ExpectIntent {
		p = append(p, fmt.Sprintf("intent=%q, want %q", r.Intent, tc.ExpectIntent))
	}
	reply, hint := strings.ToLower(r.Reply), strings.ToLower(r.UpsellHint)
	for _, s := range tc.ReplyMustNotContain {
		if strings.Contains(reply, strings.ToLower(s)) {
			p = append(p, fmt.Sprintf("reply contains %q", s))
		}
	}
	for _, s := range tc.MustNotContain {
		if strings.Contains(reply+"\n"+hint, strings.ToLower(s)) {
			p = append(p, fmt.Sprintf("reply/hint mentions %q (not in KB)", s))
		}
	}
	for _, s := range tc.HintMustContain {
		if !strings.Contains(hint, strings.ToLower(s)) {
			p = append(p, fmt.Sprintf("hint lacks %q", s))
		}
	}
	if tc.NoNewNumbers {
		// Every number in the reply and hint must come from the KB or from the customer's own text (message or history).
		allowed := slices.Concat(kbNumbers, numbers(tc.Message))
		for _, m := range tc.History {
			allowed = append(allowed, numbers(m["text"])...)
		}
		for i, text := range []string{r.Reply, r.UpsellHint} {
			for _, n := range numbers(text) {
				if !slices.Contains(allowed, n) {
					p = append(p, fmt.Sprintf("%s has number %q not in KB or customer text", []string{"reply", "hint"}[i], n))
				}
			}
		}
	}
	switch {
	case tc.ReplyLanguage == "ru" && !hasCyrillic(r.Reply):
		p = append(p, "reply is not in Russian")
	case tc.ReplyLanguage == "en" && hasCyrillic(r.Reply):
		p = append(p, "reply contains Cyrillic, want English")
	}
	// The manager hint must always be in Russian, whatever the customer's language.
	if !hasCyrillic(r.UpsellHint) {
		p = append(p, "hint is not in Russian")
	}
	return p
}

func hasCyrillic(s string) bool {
	return strings.ContainsFunc(s, func(c rune) bool { return unicode.Is(unicode.Cyrillic, c) })
}

// numbers extracts numeric tokens without thousands separators ("15 000" -> "15000").
func numbers(s string) []string {
	var out []string
	for _, n := range numberRe.FindAllString(s, -1) {
		out = append(out, strings.NewReplacer(" ", "", "\u00a0", "", "\u202f", "", ",", "").Replace(n))
	}
	return out
}

func readJSON(path string, v any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, v)
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
