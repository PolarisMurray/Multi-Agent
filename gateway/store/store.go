// Package store persists task status and run history with GORM.
//
//	Task     one row per task: its current status, read when querying a task
//	TaskRun  one row per execution (run or resume): the history, read when debugging
//
// It depends only on *gorm.DB, not on a specific database. To use MySQL:
//
//	db, err := gorm.Open(mysql.Open("user:pass@tcp(localhost:3306)/bitguide?parseTime=true"))
//	s := store.New(db)
package store

import (
	"context"
	"time"

	"gorm.io/gorm"
)

// Task statuses. Values match the workflow statuses on the Python side.
const (
	StatusQueued      = "queued"
	StatusRunning     = "running"
	StatusCompleted   = "completed"
	StatusInterrupted = "interrupted"
	StatusFailed      = "failed"
)

type Task struct {
	ID        string `gorm:"primaryKey;size:32"` // MySQL requires a length on indexed strings
	Workflow  string `gorm:"size:64;not null"`
	Input     string `gorm:"type:longtext"`
	Status    string `gorm:"size:16;not null;index"`
	Output    string `gorm:"type:longtext"`
	Interrupt string `gorm:"type:longtext"` // human approval payload, as a JSON string
	Error     string `gorm:"type:longtext"`
	Steps     int
	CreatedAt time.Time // maintained by GORM
	UpdatedAt time.Time // maintained by GORM
}

type TaskRun struct {
	ID         uint   `gorm:"primaryKey"`
	TaskID     string `gorm:"size:32;not null;index"`
	Action     string `gorm:"size:16;not null"` // run | resume
	Status     string `gorm:"size:16;not null"`
	Output     string `gorm:"type:longtext"`
	Error      string `gorm:"type:longtext"`
	Steps      int
	StartedAt  time.Time
	FinishedAt *time.Time // pointer: NULL until the run finishes
}

// Result is the outcome of one execution.
type Result struct {
	Status    string
	Output    string
	Interrupt string
	Error     string
	Steps     int
}

type Store struct {
	db *gorm.DB
}

func New(db *gorm.DB) *Store {
	return &Store{db: db}
}

// Migrate creates the tables and adds missing columns from the structs.
func (s *Store) Migrate() error {
	return s.db.AutoMigrate(&Task{}, &TaskRun{})
}

func (s *Store) CreateTask(ctx context.Context, id, workflow, input string) error {
	task := Task{ID: id, Workflow: workflow, Input: input, Status: StatusQueued}
	return s.db.WithContext(ctx).Create(&task).Error
}

// GetTask returns gorm.ErrRecordNotFound if the task does not exist.
func (s *Store) GetTask(ctx context.Context, id string) (*Task, error) {
	var task Task
	if err := s.db.WithContext(ctx).First(&task, "id = ?", id).Error; err != nil {
		return nil, err
	}
	return &task, nil
}

// StartRun adds a run record and marks the task running, in one transaction.
func (s *Store) StartRun(ctx context.Context, taskID, action string) (*TaskRun, error) {
	run := TaskRun{TaskID: taskID, Action: action, Status: StatusRunning, StartedAt: time.Now()}
	err := s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if err := tx.Create(&run).Error; err != nil {
			return err
		}
		return tx.Model(&Task{}).Where("id = ?", taskID).Update("status", StatusRunning).Error
	})
	if err != nil {
		return nil, err
	}
	return &run, nil
}

// FinishRun saves the run result and updates the task's current status.
// It updates with maps rather than structs because GORM skips zero values in
// structs: after a resume, the old interrupt and error must be cleared to "",
// and only a map can write that.
func (s *Store) FinishRun(ctx context.Context, run *TaskRun, result Result) error {
	now := time.Now()
	return s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		err := tx.Model(run).Updates(map[string]any{
			"status":      result.Status,
			"output":      result.Output,
			"error":       result.Error,
			"steps":       result.Steps,
			"finished_at": now,
		}).Error
		if err != nil {
			return err
		}
		return tx.Model(&Task{}).Where("id = ?", run.TaskID).Updates(map[string]any{
			"status":    result.Status,
			"output":    result.Output,
			"interrupt": result.Interrupt,
			"error":     result.Error,
			"steps":     result.Steps,
		}).Error
	})
}

// ListRuns returns all runs of a task in chronological order.
func (s *Store) ListRuns(ctx context.Context, taskID string) ([]TaskRun, error) {
	var runs []TaskRun
	err := s.db.WithContext(ctx).Where("task_id = ?", taskID).Order("id").Find(&runs).Error
	return runs, err
}
