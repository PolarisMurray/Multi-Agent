package main

import (
	"context"
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
	taskPrefix = "bitguide:task:"      // Hash per task: workflow, input, status, output, ...
	taskIndex  = "bitguide:task-index" // ZSET of task ids scored by creation time, for listing
	taskTopic  = "bitguide.tasks"
)

// Job is the message sent to workers; the fields match handle() in worker.py.
type Job struct {
	ID       string         `json:"id"`
	Action   string         `json:"action"` // run | resume
	Workflow string         `json:"workflow,omitempty"`
	Input    string         `json:"input,omitempty"`
	Value    map[string]any `json:"value,omitempty"` // approval decision for resume
}

// enqueue marks the task queued in Redis, then publishes the job to Kafka.
// The two writes are not atomic: if publishing fails, the task is marked failed.
func (s *server) enqueue(ctx context.Context, job Job, fields ...any) error {
	payload, err := json.Marshal(job)
	if err != nil {
		return err
	}
	key := taskPrefix + job.ID
	fields = append(fields, "status", "queued", "updated_at", now())
	if err := s.rdb.HSet(ctx, key, fields...).Err(); err != nil {
		return err
	}
	if err := s.tasks.WriteMessages(ctx, kafka.Message{Key: []byte(job.ID), Value: payload}); err != nil {
		s.rdb.HSet(ctx, key, "status", "failed", "error", err.Error())
		return err
	}
	return nil
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
	job := Job{ID: newID(), Action: "run", Workflow: req.Workflow, Input: req.Input}
	err := s.rdb.ZAdd(c, taskIndex, redis.Z{Score: float64(time.Now().UnixMilli()), Member: job.ID}).Err()
	if err == nil {
		err = s.enqueue(c, job, "workflow", req.Workflow, "input", req.Input)
	}
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusAccepted, gin.H{"id": job.ID, "status": "queued"})
}

func (s *server) list(c *gin.Context) {
	ids, err := s.rdb.ZRevRange(c, taskIndex, 0, 49).Result()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	tasks := []map[string]string{}
	for _, id := range ids {
		task, err := s.loadTask(c, id)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		if task != nil {
			tasks = append(tasks, task)
		}
	}
	c.JSON(http.StatusOK, tasks)
}

func (s *server) get(c *gin.Context) {
	task, err := s.loadTask(c, c.Param("id"))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if task == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "task not found"})
		return
	}
	c.JSON(http.StatusOK, task)
}

func (s *server) resume(c *gin.Context) {
	id := c.Param("id")
	task, err := s.loadTask(c, id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if task == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "task not found"})
		return
	}
	// Only a task waiting for approval or one that failed can continue.
	status := task["status"]
	if status != "interrupted" && status != "failed" {
		c.JSON(http.StatusConflict, gin.H{"error": "task is " + status + ", nothing to resume"})
		return
	}
	var value map[string]any
	_ = c.ShouldBindJSON(&value) // retrying a failed task needs no body
	if status == "interrupted" && value["approved"] == nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": `send {"approved": true|false, "comment": "..."}`})
		return
	}
	if err := s.enqueue(c, Job{ID: id, Action: "resume", Value: value}); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusAccepted, gin.H{"id": id, "status": "queued"})
}

// loadTask returns nil, nil when the task does not exist.
func (s *server) loadTask(ctx context.Context, id string) (map[string]string, error) {
	task, err := s.rdb.HGetAll(ctx, taskPrefix+id).Result()
	if err != nil || len(task) == 0 {
		return nil, err
	}
	task["id"] = id
	return task, nil
}

func newID() string {
	b := make([]byte, 6)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func now() string {
	return time.Now().UTC().Format(time.RFC3339)
}
