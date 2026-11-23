import { useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { Info, LoaderCircle, MessageSquareText, PanelLeftOpen, SquarePen } from 'lucide-react'
import { BrowserVerification } from '../components/BrowserVerification'
import { ChatBubble, ChatLoading, ChatWelcome } from '../components/ChatMessages'
import { Composer } from '../components/Composer'
import { ConversationHistory } from '../components/ConversationHistory'
import { DocumentSidebar } from '../components/DocumentSidebar'
import { MAX_QUESTION_LENGTH, useConversation } from '../hooks/useConversation'
import { useDemoConversation } from '../hooks/useDemoConversation'
import { useWorkspace } from '../hooks/useWorkspace'
import type { Health } from '../types/api'

type WorkspaceState = ReturnType<typeof useWorkspace>
type History = ReturnType<typeof useConversation>
type ChatView = Pick<History, 'messages' | 'draft' | 'pending' | 'error' | 'busy' | 'setDraft'>
type Mode = 'local' | 'demo' | 'connecting'

interface LayoutProps {
  workspace: WorkspaceState
  mode: Mode
  chat?: ChatView
  history?: History
  onNew?: () => Promise<boolean>
  onAsk?: () => void
  demoBlocked?: boolean
  maxQuestionLength?: number
  verification?: ReactNode
}

function ConnectionBanner({ mode, workspace }: Pick<LayoutProps, 'mode' | 'workspace'>) {
  if (mode === 'local') return null

  return (
    <div className="connection-banner" role="status">
      <Info size={17} aria-hidden="true" />
      <div className="min-w-0 flex-1">
        {mode === 'demo' && <p><strong>Public demo</strong> · Sample PDFs only. Chats are kept in this browser tab.</p>}
        {workspace.checkingHealth && <p>Connecting to the backend. A hosted demo may take a moment to wake up.</p>}
        {workspace.health === null && !workspace.checkingHealth && <p>The backend may still be waking up. Wait a moment, then try again.</p>}
      </div>
      {workspace.health === null && !workspace.checkingHealth && <button type="button" className="text-button" onClick={workspace.retry}>Try again</button>}
    </div>
  )
}

function WorkspaceLayout({ workspace, mode, chat, history, onNew, onAsk, demoBlocked, maxQuestionLength, verification }: LayoutProps) {
  const dialog = useRef<HTMLDialogElement>(null)
  const input = useRef<HTMLTextAreaElement>(null)
  const chatScroll = useRef<HTMLDivElement>(null)
  const showWelcome = !chat?.messages.length && !chat?.pending
  const knownDocumentIds = useMemo(() => mode !== 'local' || workspace.loadingDocuments || workspace.loadError
    ? undefined
    : new Set(workspace.documents.map((document) => document.id)), [mode, workspace.documents, workspace.loadError, workspace.loadingDocuments])
  const scopeCount = workspace.documentIds?.length ?? workspace.readyCount
  const scope = workspace.readyCount === 0
    ? mode === 'local' ? 'Upload a PDF to start' : 'Waiting for sample PDFs'
    : scopeCount === 0 ? 'Select a document to search'
      : workspace.documentIds === null ? `All documents · ${scopeCount} ${scopeCount === 1 ? 'PDF' : 'PDFs'}`
        : `${scopeCount} selected ${scopeCount === 1 ? 'document' : 'documents'}`

  useEffect(() => {
    if (chatScroll.current) chatScroll.current.scrollTop = chatScroll.current.scrollHeight
  }, [chat?.messages, chat?.pending])

  useEffect(() => {
    const desktop = window.matchMedia('(min-width: 768px)')
    function closeOnDesktop() {
      if (desktop.matches) dialog.current?.close()
    }
    desktop.addEventListener('change', closeOnDesktop)
    return () => desktop.removeEventListener('change', closeOnDesktop)
  }, [])

  async function newChat() {
    const created = await onNew?.() ?? false
    if (created) dialog.current?.close()
    return created
  }

  async function openChat(id: string) {
    const opened = await history?.open(id) ?? false
    if (opened) dialog.current?.close()
    return opened
  }

  const sidebarHistory = history && <ConversationHistory history={history} onNew={newChat} onOpen={openChat} />

  return (
    <div className="workspace-shell">
      <div className="desktop-sidebar">
        <DocumentSidebar workspace={workspace} mode={mode}>{sidebarHistory}</DocumentSidebar>
      </div>
      <dialog
        className="mobile-drawer"
        ref={dialog}
        aria-label="Document library"
        onClick={(event) => { if (event.target === event.currentTarget) dialog.current?.close() }}
      >
        <DocumentSidebar workspace={workspace} mode={mode} onClose={() => dialog.current?.close()}>{sidebarHistory}</DocumentSidebar>
      </dialog>

      <main className="workspace-main">
        <header className="chat-header">
          <div className="flex min-w-0 items-center gap-3">
            <button className="icon-button mobile-open" type="button" aria-label="Open document library" aria-haspopup="dialog" onClick={() => dialog.current?.showModal()}><PanelLeftOpen size={21} /></button>
            <span className="header-icon" aria-hidden="true"><MessageSquareText size={17} /></span>
            <bdi className="header-title" title={history?.active?.title}>{history?.active?.title ?? 'Document chat'}</bdi>
          </div>
          <button className="new-chat" type="button" disabled={!chat || chat.busy} onClick={() => { void newChat() }}>
            <SquarePen size={16} aria-hidden="true" /> <span>New chat</span>
          </button>
        </header>

        <ConnectionBanner mode={mode} workspace={workspace} />

        <div className="chat-scroll" ref={chatScroll}>
          {!chat || history?.restoring ? (
            <p className="conversation-loading" role="status">
              {(workspace.checkingHealth || history?.restoring) && <LoaderCircle size={18} className="spin" aria-hidden="true" />}
              {!chat ? workspace.checkingHealth ? 'Getting the workspace ready…' : 'Workspace unavailable' : 'Loading conversation…'}
            </p>
          ) : showWelcome ? (
            <ChatWelcome demo={mode === 'demo'} onSuggestion={(question) => { chat.setDraft(question.slice(0, maxQuestionLength)); input.current?.focus() }} />
          ) : (
            <div className="chat-log" role="log" aria-label="Conversation" aria-relevant="additions">
              {chat.messages.map((message) => <ChatBubble key={message.id} message={message} knownDocumentIds={knownDocumentIds} />)}
              {chat.pending && <><ChatBubble message={chat.pending} /><ChatLoading /></>}
            </div>
          )}
        </div>

        <Composer
          input={input}
          draft={chat?.draft ?? ''}
          onDraftChange={(draft) => chat?.setDraft(draft)}
          onSend={() => onAsk?.()}
          pending={Boolean(chat?.pending)}
          disabled={!chat || chat.busy}
          scope={scope}
          canSearch={scopeCount > 0 && !demoBlocked}
          error={chat?.error}
          maxQuestionLength={maxQuestionLength}
          verification={verification}
        />
      </main>
    </div>
  )
}

function LocalWorkspace({ workspace }: { workspace: WorkspaceState }) {
  const history = useConversation(workspace.restoreScope)
  return <WorkspaceLayout workspace={workspace} mode="local" history={history} chat={history}
    onNew={() => history.create(workspace.documentIds)} onAsk={() => { void history.send(workspace.documentIds) }} />
}

function DemoWorkspace({ workspace, config }: { workspace: WorkspaceState; config: Health }) {
  const maxQuestionLength = config.max_question_length ?? MAX_QUESTION_LENGTH
  const chat = useDemoConversation(maxQuestionLength)
  const [token, setToken] = useState<string>()
  const [challengeCycle, setChallengeCycle] = useState(0)
  const siteKey = config.turnstile_site_key

  function send() {
    if (siteKey && !token) return
    const sent = chat.send(workspace.documentIds, token)
    if (siteKey) {
      setToken(undefined)
      // A token may be consumed even when request processing subsequently fails.
      void sent.finally(() => setChallengeCycle((current) => current + 1))
    }
  }

  return <WorkspaceLayout workspace={workspace} mode="demo" chat={chat} onNew={async () => chat.clear()} onAsk={send}
    maxQuestionLength={maxQuestionLength} demoBlocked={chat.rateLimited || Boolean(siteKey && !token)}
    verification={siteKey && <BrowserVerification siteKey={siteKey} action={config.turnstile_action ?? 'demo_chat'} cycle={challengeCycle} onToken={setToken} />} />
}

export function Workspace() {
  const workspace = useWorkspace()
  if (!workspace.backendConfig) return <WorkspaceLayout workspace={workspace} mode="connecting" />
  if (workspace.backendConfig.demo_mode) return <DemoWorkspace workspace={workspace} config={workspace.backendConfig} />
  return <LocalWorkspace workspace={workspace} />
}
