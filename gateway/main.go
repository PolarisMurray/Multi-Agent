// Gin gateway: accepts workflow tasks, hands them to Python workers through Kafka,
// and streams their progress back to the browser over SSE.
//
//	browser ──HTTP──► gateway ──bitguide.tasks──► worker
//	browser ◄──SSE─── gateway ◄─bitguide.events── worker
//
// Redis holds the current state of every task, and only the gateway writes it.
package main

import (
	"context"
	"log"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
	"github.com/segmentio/kafka-go"
)

type server struct {
	rdb   *redis.Client
	tasks *kafka.Writer
	hub   *hub
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	brokers := strings.Split(getenv("KAFKA_BROKERS", "localhost:9092"), ",")
	s := &server{
		rdb: redis.NewClient(&redis.Options{Addr: getenv("REDIS_ADDR", "localhost:6379")}),
		tasks: &kafka.Writer{
			Addr:         kafka.TCP(brokers...),
			Topic:        taskTopic,
			Balancer:     &kafka.Hash{}, // partition by key (task id): one task's messages stay in order
			RequiredAcks: kafka.RequireAll,
			BatchTimeout: 10 * time.Millisecond, // default batches for 1s, which delays every request
		},
		hub: newHub(),
	}
	go s.consumeEvents(context.Background(), brokers)

	r := gin.Default()
	r.POST("/tasks", s.submit)
	r.GET("/tasks", s.list)
	r.GET("/tasks/:id", s.get)
	r.POST("/tasks/:id/resume", s.resume)
	r.GET("/tasks/:id/events", s.events)
	if err := r.Run(getenv("ADDR", ":8080")); err != nil {
		log.Fatal(err)
	}
}
