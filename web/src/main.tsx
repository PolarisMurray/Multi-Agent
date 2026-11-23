import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, NavLink, Route, Routes } from 'react-router'
import SubmitPage from './pages/SubmitPage'
import TaskListPage from './pages/TaskListPage'
import TaskDetailPage from './pages/TaskDetailPage'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <nav>
        <strong>BitGuide</strong>
        <NavLink to="/" end>提交任务</NavLink>
        <NavLink to="/tasks" end>任务列表</NavLink>
      </nav>
      <main>
        <Routes>
          <Route path="/" element={<SubmitPage />} />
          <Route path="/tasks" element={<TaskListPage />} />
          <Route path="/tasks/:id" element={<TaskDetailPage />} />
        </Routes>
      </main>
    </BrowserRouter>
  </StrictMode>,
)
