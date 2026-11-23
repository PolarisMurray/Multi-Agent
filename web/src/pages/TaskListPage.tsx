import { useEffect, useState } from 'react'
import { Link } from 'react-router'
import { api, STATUS_LABELS, type Task } from '../api'

export default function TaskListPage() {
  const [tasks, setTasks] = useState<Task[]>([])
  const [error, setError] = useState('')

  function load() {
    api.list().then(setTasks, (err) => setError(String(err)))
  }
  useEffect(load, [])

  return (
    <>
      <h1>任务列表 <button onClick={load}>刷新</button></h1>
      {error && <p className="error">{error}</p>}
      <table>
        <thead>
          <tr><th>ID</th><th>工作流</th><th>输入</th><th>状态</th><th>更新时间</th></tr>
        </thead>
        <tbody>
          {tasks.map((t) => (
            <tr key={t.id}>
              <td><Link to={`/tasks/${t.id}`}>{t.id}</Link></td>
              <td>{t.workflow}</td>
              <td>{t.input}</td>
              <td><span className={`status ${t.status}`}>{STATUS_LABELS[t.status] ?? t.status}</span></td>
              <td>{t.updated_at}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}
