import { useRef, useState } from 'react'
import type { DragEvent, ReactNode } from 'react'
import { ArrowUpFromLine, CircleAlert, FileText, Files, Layers, LoaderCircle, RefreshCw, Trash2, X } from 'lucide-react'
import type { useWorkspace } from '../hooks/useWorkspace'

type Workspace = ReturnType<typeof useWorkspace>

interface DocumentSidebarProps {
  workspace: Workspace
  onClose?: () => void
  children?: ReactNode
  mode: 'local' | 'demo' | 'connecting'
}

function WorkspaceStatus({ health, retry, mode }: Pick<Workspace, 'health' | 'retry'> & Pick<DocumentSidebarProps, 'mode'>) {
  const status = health === undefined ? 'Connecting' : health === null ? 'Backend offline'
    : health.groq_configured ? 'Ready' : mode === 'demo' ? 'Demo unavailable' : 'Groq key needed'

  return (
    <div className="workspace-status">
      <span className={`status-dot ${health?.groq_configured ? 'status-dot-ready' : ''}`} />
      <div className="min-w-0 flex-1">
        <span className="status-label">RAG system</span>
        <span className="status-value" role="status">{status}</span>
      </div>
      {health !== undefined && !health?.groq_configured && (
        <button className="icon-button" type="button" aria-label="Refresh system status" onClick={retry}>
          <RefreshCw size={15} />
        </button>
      )}
    </div>
  )
}

export function DocumentSidebar({ workspace, onClose, children, mode }: DocumentSidebarProps) {
  const input = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const dragDepth = useRef(0)
  const uploading = workspace.mutation?.type === 'upload' ? workspace.mutation : null
  const busy = workspace.mutation !== undefined
  const scoped = workspace.documentIds
  const editable = mode === 'local'

  function onDragEnter(event: DragEvent) {
    if (!event.dataTransfer.types.includes('Files')) return
    event.preventDefault()
    dragDepth.current += 1
    if (!busy) setDragging(true)
  }

  function onDragLeave(event: DragEvent) {
    if (!event.dataTransfer.types.includes('Files')) return
    dragDepth.current = Math.max(0, dragDepth.current - 1)
    if (!dragDepth.current) setDragging(false)
  }

  function onDragOver(event: DragEvent) {
    if (!event.dataTransfer.types.includes('Files')) return
    event.preventDefault()
    event.dataTransfer.dropEffect = busy ? 'none' : 'copy'
  }

  function onDrop(event: DragEvent) {
    if (!event.dataTransfer.types.includes('Files')) return
    event.preventDefault()
    dragDepth.current = 0
    setDragging(false)
    if (!busy) void workspace.upload(Array.from(event.dataTransfer.files))
  }

  return (
    <aside className="document-sidebar" aria-label="Document library">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true"><Files size={21} strokeWidth={1.9} /></span>
        <div className="min-w-0 flex-1">
          <p className="brand-name">AskMyDocs <span>AI</span></p>
          <p className="brand-description">Document workspace</p>
        </div>
        {onClose && (
          <button type="button" className="icon-button" aria-label="Close document library" onClick={onClose}><X size={19} /></button>
        )}
      </div>

      {children}

      <div className="sidebar-body">
        {editable && <div
          className={`upload-drop${dragging ? ' upload-drop-active' : ''}`}
          onDragEnter={onDragEnter}
          onDragLeave={onDragLeave}
          onDragOver={onDragOver}
          onDrop={onDrop}
        >
          <input
            type="file"
            ref={input}
            accept=".pdf,application/pdf"
            multiple
            hidden
            onChange={(event) => {
              const files = Array.from(event.currentTarget.files ?? [])
              event.currentTarget.value = ''
              void workspace.upload(files)
            }}
          />
          <button type="button" className="upload-button" disabled={busy} onClick={() => input.current?.click()}>
            <span className="upload-icon" aria-hidden="true">
              {uploading ? <LoaderCircle size={19} className="spin" /> : <ArrowUpFromLine size={19} />}
            </span>
            <span className="upload-title">
              {uploading ? `Indexing ${uploading.count} ${uploading.count === 1 ? 'PDF' : 'PDFs'}…` : 'Upload PDFs'}
            </span>
            <span className="upload-hint">{uploading ? 'First upload may take a moment' : 'Choose files or drag them here'}</span>
          </button>
        </div>}

        {editable && (workspace.issues.length > 0 || workspace.actionError) && (
          <div className="sidebar-alert" role="alert">
            <div className="flex items-start justify-between gap-2">
              <p className="flex items-center gap-2 font-semibold"><CircleAlert size={15} /> {workspace.actionError ? 'Document action failed' : 'Some files could not be indexed'}</p>
              <button type="button" className="icon-button small-icon-button" aria-label="Dismiss document errors" onClick={workspace.dismissNotices}><X size={14} /></button>
            </div>
            {workspace.actionError && <p>{workspace.actionError}</p>}
            {workspace.issues.map((issue, index) => (
              <p key={`${issue.filename}-${index}`}><bdi>{issue.filename}</bdi> — {issue.message}</p>
            ))}
          </div>
        )}
        <p className="sr-only" aria-live="polite">{uploading ? `Indexing ${uploading.count} PDF files.` : workspace.statusMessage}</p>

        <div className="documents-heading">
          <h2>{mode === 'demo' ? 'Sample PDFs' : 'Documents'}</h2>
          <span className="documents-count" aria-label={`${workspace.documents.length} documents`}>{workspace.documents.length}</span>
        </div>

        {workspace.readyCount > 0 && (
          <button type="button" className="all-documents" aria-pressed={scoped === null} onClick={workspace.selectAll}>
            <Layers size={16} aria-hidden="true" />
            <span className="flex-1 text-left">All documents</span>
            <span className="scope-indicator" aria-hidden="true" />
          </button>
        )}

        {workspace.loadError && (
          <div className="library-state">
            <CircleAlert size={18} aria-hidden="true" />
            <p>Could not load the library.</p>
            <button type="button" className="text-button" onClick={workspace.retry}>Try again</button>
          </div>
        )}

        {workspace.documents.length === 0 && !workspace.loadError && (
          <div className="library-state">
            {workspace.loadingDocuments ? <LoaderCircle size={19} className="spin" aria-hidden="true" /> : <FileText size={19} aria-hidden="true" />}
            <p>{workspace.loadingDocuments ? 'Loading your library…' : 'No PDFs yet'}</p>
            {!workspace.loadingDocuments && <span>{mode === 'demo' ? 'Sample PDFs are currently unavailable.' : 'Indexed files appear here.'}</span>}
          </div>
        )}

        {workspace.documents.length > 0 && (
          <ul className="document-list" aria-label="Uploaded PDFs">
            {workspace.documents.map((document) => {
              const isDeleting = document.status === 'deleting' || (workspace.mutation?.type === 'delete' && workspace.mutation.id === document.id)
              const isReady = document.status === 'ready'
              const selected = scoped?.includes(document.id) ?? false
              const allowSelection = isReady && !isDeleting && (selected || (scoped?.length ?? 0) < 100)

              return (
                <li className={`document-row${selected ? ' document-row-selected' : ''}`} key={document.id}>
                  <label className="document-file" title={document.filename}>
                    <input
                      className="document-checkbox"
                      type="checkbox"
                      checked={selected}
                      disabled={!allowSelection}
                      onChange={() => workspace.toggleDocument(document.id)}
                      aria-label={`Select ${document.filename} for search`}
                    />
                    <span className="document-file-text">
                      <bdi className="document-name">{document.filename}</bdi>
                      <span className="document-meta">
                        {isDeleting ? 'Deleting' : isReady ? `${document.page_count} ${document.page_count === 1 ? 'page' : 'pages'} · Indexed` : 'Processing'}
                      </span>
                    </span>
                  </label>
                  {editable && <button
                    type="button"
                    className="icon-button delete-button"
                    aria-label={`Delete ${document.filename}`}
                    title={`Delete ${document.filename}`}
                    disabled={busy || document.status === 'processing'}
                    onClick={() => { void workspace.remove(document) }}
                  >
                    {workspace.mutation?.type === 'delete' && workspace.mutation.id === document.id ? <LoaderCircle size={15} className="spin" /> : <Trash2 size={15} />}
                  </button>}
                </li>
              )
            })}
          </ul>
        )}
      </div>

      <WorkspaceStatus health={workspace.health} retry={workspace.retry} mode={mode} />
    </aside>
  )
}
