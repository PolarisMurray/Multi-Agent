import { useEffect, useState } from 'react'
import { useParams } from 'react-router'
import { api, STATUS_LABELS, type Task, type TaskEvent } from '../api'

export default function TaskDetailPage() {
  const { id = '' } = useParams()
  const [task, setTask] = useState<Task | null>(null)
  const [log, setLog] = useState<string[]>([])
  const [error, setError] = useState('')

  // One SSE connection per page: a "snapshot" of the current state, then live "event"s.
  // EventSource reconnects by itself and gets a fresh snapshot each time.
  useEffect(() => {
    const source = new EventSource(api.eventsUrl(id))
    source.onopen = () => setError('')
    source.onerror = () =>
      setError(source.readyState === EventSource.CLOSED ? '连接失败：任务不存在或网关未启动' : '连接断开，正在重连…')
    source.addEventListener('snapshot', (e) => setTask(JSON.parse(e.data)))
    source.addEventListener('event', (e) => {
      const event: TaskEvent = JSON.parse(e.data)
      setLog((lines) => [...lines, describe(event)])
      if (event.type === 'status') {
        setTask((t) => t && {
          ...t,
          status: event.status,
          steps: String(event.steps ?? 0),
          output: event.output ?? '',
          interrupt: event.interrupt ?? '',
          error: event.error ?? '',
        })
      }
    })
    return () => source.close()
  }, [id])

  if (!task) return <p className={error ? 'error' : ''}>{error || '加载中…'}</p>

  return (
    <>
      <h1>
        任务 {task.id} <span className={`status ${task.status}`}>{STATUS_LABELS[task.status] ?? task.status}</span>
      </h1>
      {error && <p className="error">{error}</p>}
      <p>工作流：{task.workflow}　输入：{task.input}　步数：{task.steps || 0}</p>

      {task.status === 'interrupted' && task.interrupt && (
        <Approval id={task.id} interrupt={task.interrupt} onSent={() => setTask({ ...task, status: 'queued' })} />
      )}
      {task.output && <><h2>结果</h2><pre>{task.output}</pre></>}
      {task.error && <><h2>错误</h2><pre className="error">{task.error}</pre></>}

      <h2>实时进度</h2>
      <ol className="log">{log.map((line, i) => <li key={i}>{line}</li>)}</ol>
    </>
  )
}

function Approval({ id, interrupt, onSent }: { id: string; interrupt: string; onSent: () => void }) {
  const [comment, setComment] = useState('')
  const [error, setError] = useState('')
  const payload = JSON.parse(interrupt).payload ?? {}

  async function decide(approved: boolean) {
    try {
      await api.resume(id, approved, comment)
      onSent()
    } catch (err) {
      setError(String(err))
    }
  }

  return (
    <section className="approval">
      <h2>{payload.question ?? '等待人工审批'}</h2>
      {payload.preview && <pre>{payload.preview}</pre>}
      <input value={comment} onChange={(e) => setComment(e.target.value)} placeholder="审批意见（驳回时会交给 Agent 修改）" />
      <button onClick={() => decide(true)}>通过</button>
      <button className="secondary" onClick={() => decide(false)}>驳回</button>
      {error && <p className="error">{error}</p>}
    </section>
  )
}

function describe(event: TaskEvent): string {
  if (event.type === 'status') return `状态 → ${STATUS_LABELS[event.status] ?? event.status}`
  if (event.kind === 'workflow.step') return `第 ${event.step} 步：${event.nodes.join('、')}`
  return `  ✓ ${event.node} 完成`
}
