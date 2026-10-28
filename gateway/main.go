package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
	"github.com/segmentio/kafka-go"
)

const (
	taskPrefix = "bitguide:task:"
	taskTopic  = "bitguide.tasks"
)

type Job struct {
	ID       string
	Action   string
	Workflow string
	Input    string
	Value    map[string]any
}
type server struct {
	rdb *redis.Client
	kw  *kafaka.Writer
}

func (s *server) enqueue(c *gin.Context, job Job, fields ...any) error {
	payload, err := json.Marshal(job)
	if err != nil {
		return err
	}

	fields = append(fields, "status", "queued", "updated_at", time.Now().UTC().Format((time.RFC3339)))
	if err := s.rdb.HSet(c, taskPrefix+job.ID, fields...).Err(); err != nil {
		return err
	}
	err = s.kw.WriteMessages(c, kafka.Message{Key: []byte(job.ID), Value: payload})
	if err != nil {
		s.rdb.HSet(c, taskPrefix+job.ID, "status", "failed", "error", err.Error())
	}
	return err
}
func (s *server) submit(c *gin.Context) {
	var req struct {
		Workflow string `json:"workflow" binding:"required"`
		Input    string `json:"input" binding:"required"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	id := newID()
	err := s.rdb.HSet(c, taskPrefix+id,
		"workflow", req.Workflow,
		"input", req.Input,
		"status", "queued",
		"updated_at", time.Now().UTC().Format(time.RFC3339),
	).Err()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusAccepted, gin.H{"id": id, "status": "queued"})
}

func (s *server) get(c *gin.Context) {
	task, err := s.rdb.HGetAll(c, taskPrefix+c.Param("id")).Result()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if len(task) == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "task not found"})
		return
	}
	task["id"] = c.Param("id")
	c.JSON(http.StatusOK, task)
}
func newID() string {
	b := make([]byte, 6)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func main() {
	s := &server{
		rdb: redis.NewClient(&redis.Options{
			Addr: "localhost:6379",
		}),
	}
	r := gin.Default()
	r.GET("/ping", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"message": "pong"})
	})
	r.POST("/tasks", s.submit)
	r.GET("/tasks/:id", s.get)

	r.Run(":8080")
}
