import { useState } from 'react'
import { Check, CircleAlert, LoaderCircle, MessageCircle, Pencil, Plus, Trash2, X } from 'lucide-react'
import { MAX_TITLE_LENGTH } from '../hooks/useConversation'
import type { useConversation } from '../hooks/useConversation'
import type { ConversationSummary } from '../types/api'

type History = ReturnType<typeof useConversation>
type Editor = { type: 'rename'; id: string; title: string } | { type: 'delete'; id: string }

const groupNames = ['Today', 'Yesterday', 'Previous 7 days', 'Older']

function groupedConversations(conversations: ConversationSummary[]) {
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  const yesterday = new Date(today)
  yesterday.setDate(today.getDate() - 1)
  const week = new Date(today)
  week.setDate(today.getDate() - 7)
  const groups: Record<string, ConversationSummary[]> = Object.fromEntries(groupNames.map((name) => [name, []]))
  for (const conversation of conversations) {
    const updated = new Date(conversation.updated_at)
    const group = updated >= today ? 'Today' : updated >= yesterday ? 'Yesterday' : updated >= week ? 'Previous 7 days' : 'Older'
    groups[group]?.push(conversation)
  }
  return Object.entries(groups).filter(([, entries]) => entries.length > 0)
}

interface ConversationHistoryProps {
  history: History
  onNew: () => Promise<boolean>
  onOpen: (id: string) => Promise<boolean>
}

export function ConversationHistory({ history, onNew, onOpen }: ConversationHistoryProps) {
  const [editor, setEditor] = useState<Editor>()
  const groups = groupedConversations(history.conversations)

  async function saveTitle(edit: Extract<Editor, { type: 'rename' }>) {
    if (await history.rename(edit.id, edit.title)) setEditor(undefined)
  }

  async function deleteChat(id: string) {
    if (await history.remove(id)) setEditor(undefined)
  }

  return (
    <section className="conversation-history" aria-label="Conversation history">
      <div className="history-heading">
        <h2>Conversations</h2>
        <button type="button" className="history-new" disabled={history.busy} onClick={() => { void onNew() }}>
          <Plus size={14} aria-hidden="true" /> New chat
        </button>
      </div>

      {history.historyError && (
        <div className="history-error" role="alert">
          <CircleAlert size={14} aria-hidden="true" />
          <div><p>{history.historyError}</p><button type="button" className="text-button" onClick={history.retry}>Refresh history</button></div>
        </div>
      )}

      {history.restoring && <p className="history-status" role="status"><LoaderCircle size={14} className="spin" /> Loading conversations…</p>}
      {history.operation === 'opening' && <p className="history-status" role="status"><LoaderCircle size={14} className="spin" /> Opening conversation…</p>}
      {history.operation === 'creating' && <p className="history-status" role="status"><LoaderCircle size={14} className="spin" /> Starting conversation…</p>}
      {!history.restoring && !history.historyError && !groups.length && <p className="history-status">No saved chats yet</p>}

      <div className="history-list">
        {groups.map(([label, conversations]) => (
          <div className="history-group" key={label}>
            <h3>{label}</h3>
            <ul aria-label={label}>
              {conversations.map((conversation) => {
                const selected = history.active?.id === conversation.id
                const edit = editor?.id === conversation.id ? editor : undefined
                const available = !history.busy

                if (edit?.type === 'rename') {
                  return (
                    <li className="history-row" key={conversation.id}>
                      <form className="history-edit" onSubmit={(event) => { event.preventDefault(); void saveTitle(edit) }}>
                        <input
                          autoFocus
                          dir="auto"
                          value={edit.title}
                          aria-label="Conversation title"
                          maxLength={MAX_TITLE_LENGTH}
                          disabled={!available}
                          onChange={(event) => setEditor({ ...edit, title: event.currentTarget.value })}
                          onKeyDown={(event) => { if (event.key === 'Escape') { event.preventDefault(); setEditor(undefined) } }}
                        />
                        <button className="icon-button small-icon-button" type="submit" aria-label="Save conversation title" disabled={!available || !edit.title.trim()}><Check size={15} /></button>
                        <button className="icon-button small-icon-button" type="button" aria-label="Cancel renaming" disabled={!available} onClick={() => setEditor(undefined)}><X size={15} /></button>
                      </form>
                    </li>
                  )
                }

                if (edit?.type === 'delete') {
                  return (
                    <li className="history-row history-confirm" key={conversation.id}>
                      <span>Delete this chat?</span>
                      <button type="button" className="history-delete" disabled={!available} onClick={() => { void deleteChat(conversation.id) }} aria-label={`Confirm delete ${conversation.title}`}>Delete</button>
                      <button type="button" className="icon-button small-icon-button" aria-label="Cancel deleting" disabled={!available} onClick={() => setEditor(undefined)}><X size={15} /></button>
                    </li>
                  )
                }

                return (
                  <li className={`history-row${selected ? ' history-row-selected' : ''}`} key={conversation.id}>
                    <button className="history-open" type="button" disabled={!available} aria-current={selected ? 'true' : undefined} title={conversation.title} onClick={() => { void onOpen(conversation.id) }}>
                      <MessageCircle size={14} aria-hidden="true" /> <bdi>{conversation.title}</bdi>
                    </button>
                    <button className="icon-button small-icon-button" type="button" disabled={!available} title="Rename conversation" aria-label={`Rename ${conversation.title}`} onClick={() => setEditor({ type: 'rename', id: conversation.id, title: conversation.title })}>
                      <Pencil size={13} />
                    </button>
                    <button className="icon-button small-icon-button history-remove" type="button" disabled={!available} title="Delete conversation" aria-label={`Delete ${conversation.title}`} onClick={() => setEditor({ type: 'delete', id: conversation.id })}>
                      <Trash2 size={13} />
                    </button>
                  </li>
                )
              })}
            </ul>
          </div>
        ))}
      </div>
    </section>
  )
}
