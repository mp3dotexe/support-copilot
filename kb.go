package main

import (
	"encoding/json"
	"errors"
	"os"
)

// Entry is one knowledge base article. Replace knowledge_base.json with real data.
type Entry struct {
	ID              string   `json:"id"`
	Topic           string   `json:"topic"`
	Answer          string   `json:"answer"`
	RelatedProducts []string `json:"related_products"`
}

func loadKB(path string) ([]Entry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var kb []Entry
	if err := json.Unmarshal(data, &kb); err != nil {
		return nil, err
	}
	if len(kb) == 0 {
		return nil, errors.New("knowledge base is empty")
	}
	return kb, nil
}
