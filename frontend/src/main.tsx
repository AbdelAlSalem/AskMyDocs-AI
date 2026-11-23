import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Workspace } from './pages/Workspace'
import './styles.css'

const root = document.getElementById('root')
if (!root) throw new Error('Could not mount AskMyDocs AI.')

createRoot(root).render(<StrictMode><Workspace /></StrictMode>)
