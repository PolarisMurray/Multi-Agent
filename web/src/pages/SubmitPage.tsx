import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router'
import { api, WORKFLOWS } from '../api'

export default function SubmitPage() {
  const [workflow, setWorkflow] = useState(WORKFLOWS[0])
  const [input, setInput] = useState('')
  const [error, setError] = useState('')
  const navigate = useNavigate()

  async function submit(e: FormEvent) {
    e.preventDefault()
    try {
      const { id } = await api.submit(workflow, input)
      navigate(`/tasks/${id}`) // go straight to the live view
    } catch (err) {
      setError(String(err))
    }
  }

  return (
    <form onSubmit={submit}>
      <h1>提交任务</h1>
      <label>
        工作流
        <select value={workflow} onChange={(e) => setWorkflow(e.target.value)}>
          {WORKFLOWS.map((name) => <option key={name}>{name}</option>)}
        </select>
      </label>
      <label>
        输入
        <textarea rows={4} value={input} onChange={(e) => setInput(e.target.value)} placeholder="例如：周五团建" required />
      </label>
      <button type="submit">提交</button>
      {error && <p className="error">{error}</p>}
    </form>
  )
}
