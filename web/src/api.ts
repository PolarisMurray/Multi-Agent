// Task as stored by the gateway in Redis: every field is a string.
export type Task = {
  id: string
  workflow: string
  input: string
  status: string
  steps?: string
  output?: string
  interrupt?: string // JSON: {"node": "...", "payload": {"question": "...", "preview": "..."}}
  error?: string
  updated_at: string
}

// Events the worker sends back through Kafka; the gateway forwards them over SSE.
export type TaskEvent =
  | { type: 'status'; status: string; steps?: number; output?: string | null; interrupt?: string | null; error?: string | null }
  | { type: 'progress'; kind: 'workflow.step'; step: number; nodes: string[] }
  | { type: 'progress'; kind: 'workflow.node_done'; node: string }

export const WORKFLOWS = [
  'human_approval', 'expert_review', 'parallel_experts', 'pipeline',
  'review_loop', 'router', 'supervisor', 'tool_pipeline',
]

export const STATUS_LABELS: Record<string, string> = {
  queued: '排队中', running: '运行中', interrupted: '等待审批', completed: '已完成', failed: '失败',
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, { headers: { 'Content-Type': 'application/json' }, ...init })
  const body = await res.json()
  if (!res.ok) throw new Error(body.error ?? res.statusText)
  return body
}

export const api = {
  submit: (workflow: string, input: string) =>
    request<{ id: string }>('/tasks', { method: 'POST', body: JSON.stringify({ workflow, input }) }),
  list: () => request<Task[]>('/tasks'),
  resume: (id: string, approved: boolean, comment: string) =>
    request(`/tasks/${id}/resume`, { method: 'POST', body: JSON.stringify({ approved, comment }) }),
  eventsUrl: (id: string) => `/api/tasks/${id}/events`,
}
