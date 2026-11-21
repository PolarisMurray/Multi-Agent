package main

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"strconv"
	"sync"

	"github.com/gin-gonic/gin"
	"github.com/segmentio/kafka-go"
)

const eventTopic = "bitguide.events"

// Event is what workers publish. "status" events change the task's state;
// "progress" events (workflow steps) are only forwarded to the browser.
type Event struct {
	ID        string `json:"id"`
	Type      string `json:"type"` // status | progress
	Status    string `json:"status"`
	Steps     int    `json:"steps"`
	Output    string `json:"output"`
	Interrupt string `json:"interrupt"` // JSON string, set when waiting for approval
	Error     string `json:"error"`
}

// consumeEvents applies worker events to Redis and pushes them to SSE clients.
// The offset is committed after handling, so a crash replays events instead of losing them.
func (s *server) consumeEvents(ctx context.Context, brokers []string) {
	r := kafka.NewReader(kafka.ReaderConfig{Brokers: brokers, Topic: eventTopic, GroupID: "bitguide-gateway"})
	defer r.Close()
	for {
		m, err := r.FetchMessage(ctx)
		if err != nil {
			log.Printf("events: %v", err)
			return
		}
		var ev Event
		if err := json.Unmarshal(m.Value, &ev); err != nil {
			log.Printf("events: bad message: %v", err)
		} else {
			if ev.Type == "status" {
				err := s.rdb.HSet(ctx, taskPrefix+ev.ID,
					"status", ev.Status,
					"steps", strconv.Itoa(ev.Steps),
					"output", ev.Output,
					"interrupt", ev.Interrupt,
					"error", ev.Error,
					"updated_at", now(),
				).Err()
				if err != nil {
					log.Printf("events: save %s: %v", ev.ID, err)
				}
			}
			s.hub.publish(ev.ID, m.Value)
		}
		if err := r.CommitMessages(ctx, m); err != nil {
			log.Printf("events: commit: %v", err)
		}
	}
}

// events streams one task over SSE: a "snapshot" of the current state first,
// then every worker event for that task as it arrives.
func (s *server) events(c *gin.Context) {
	id := c.Param("id")
	ch := s.hub.subscribe(id) // subscribe before reading the snapshot so nothing falls in between
	defer s.hub.unsubscribe(id, ch)

	task, err := s.loadTask(c, id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if task == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "task not found"})
		return
	}
	c.SSEvent("snapshot", task)
	c.Writer.Flush()
	c.Stream(func(w io.Writer) bool {
		select {
		case msg := <-ch:
			c.SSEvent("event", json.RawMessage(msg))
			return true
		case <-c.Request.Context().Done(): // browser closed the page
			return false
		}
	})
}

// hub fans worker events out to the SSE connections watching each task.
type hub struct {
	mu   sync.Mutex
	subs map[string]map[chan []byte]bool
}

func newHub() *hub {
	return &hub{subs: map[string]map[chan []byte]bool{}}
}

func (h *hub) subscribe(id string) chan []byte {
	ch := make(chan []byte, 16)
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.subs[id] == nil {
		h.subs[id] = map[chan []byte]bool{}
	}
	h.subs[id][ch] = true
	return ch
}

func (h *hub) unsubscribe(id string, ch chan []byte) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.subs[id], ch)
	if len(h.subs[id]) == 0 {
		delete(h.subs, id)
	}
}

func (h *hub) publish(id string, msg []byte) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for ch := range h.subs[id] {
		select {
		case ch <- msg:
		default: // a stuck browser must not block the Kafka consumer; it can reload
		}
	}
}
