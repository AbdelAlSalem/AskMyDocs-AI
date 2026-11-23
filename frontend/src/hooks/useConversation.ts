import { useEffect, useMemo, useRef, useState } from 'react'
import { askQuestion, createConversation, deleteConversation, errorMessage, getConversation, listConversations, renameConversation } from '../services/api'
import type { ChatMessage, Conversation, ConversationSummary } from '../types/api'

export const MAX_QUESTION_LENGTH = 2_000
export const MAX_TITLE_LENGTH = 80
const ACTIVE_KEY = 'askmydocs:active-conversation:v1'

function readActiveId(): string | null {
  try {
    return localStorage.getItem(ACTIVE_KEY)
  } catch {
    return null
  }
}

function rememberActiveId(id?: string) {
  try {
    if (id) localStorage.setItem(ACTIVE_KEY, id)
    else localStorage.removeItem(ACTIVE_KEY)
  } catch {
    // The conversation list remains available when browser storage is disabled.
  }
}

export function useConversation(restoreScope: (ids: string[] | null) => void) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [active, setActive] = useState<Conversation | null>(null)
  const [restoring, setRestoring] = useState(true)
  const [operation, setOperation] = useState<'creating' | 'opening' | 'renaming' | 'deleting' | 'asking'>()
  const [refresh, setRefresh] = useState(0)
  const [historyError, setHistoryError] = useState<string>()
  const [draft, setDraft] = useState('')
  const [pending, setPending] = useState<ChatMessage>()
  const [error, setError] = useState<string>()
  const requestInProgress = useRef(false)
  const storedMessages = active?.messages
  const messages = useMemo<ChatMessage[]>(() => (storedMessages ?? []).map((message) => ({
    id: message.id,
    role: message.role,
    content: message.content,
    sources: message.source_references ?? undefined,
  })), [storedMessages])

  useEffect(() => {
    const controller = new AbortController()
    const signal = controller.signal
    setRestoring(true)
    setHistoryError(undefined)

    async function restore() {
      try {
        const summaries = await listConversations(signal)
        if (signal.aborted) return
        setConversations(summaries)
        const rememberedId = readActiveId()
        const previous = summaries.find(({ id }) => id === rememberedId) ?? summaries[0]
        const conversation = previous ? await getConversation(previous.id, signal) : null
        if (signal.aborted) return
        setActive(conversation)
        rememberActiveId(conversation?.id)
        if (conversation) restoreScope(conversation.document_ids)
      } catch (failure) {
        if (!signal.aborted) setHistoryError(errorMessage(failure))
      } finally {
        if (!signal.aborted) setRestoring(false)
      }
    }

    void restore()
    return () => controller.abort()
  }, [refresh, restoreScope])

  function updateSummary(summary: ConversationSummary) {
    setConversations((current) => [summary, ...current.filter(({ id }) => id !== summary.id)])
  }

  function activate(conversation: Conversation | null) {
    setActive(conversation)
    rememberActiveId(conversation?.id)
    if (conversation) restoreScope(conversation.document_ids)
    setDraft('')
    setError(undefined)
  }

  async function manage(kind: NonNullable<typeof operation>, action: () => Promise<void>): Promise<boolean> {
    if (restoring || requestInProgress.current) return false
    requestInProgress.current = true
    setOperation(kind)
    setHistoryError(undefined)
    try {
      await action()
      return true
    } catch (failure) {
      if (kind === 'asking') setError(errorMessage(failure))
      else setHistoryError(errorMessage(failure))
      return false
    } finally {
      requestInProgress.current = false
      setOperation(undefined)
    }
  }

  function create(documentIds: string[] | null) {
    return manage('creating', async () => {
      const conversation = await createConversation(documentIds)
      updateSummary(conversation)
      activate(conversation)
    })
  }

  function open(id: string) {
    if (id === active?.id) return Promise.resolve(true)
    return manage('opening', async () => { activate(await getConversation(id)) })
  }

  function rename(id: string, title: string) {
    const cleanedTitle = title.trim()
    if (!cleanedTitle || cleanedTitle.length > MAX_TITLE_LENGTH) return Promise.resolve(false)
    return manage('renaming', async () => {
      const summary = await renameConversation(id, cleanedTitle)
      updateSummary(summary)
      setActive((current) => current?.id === id ? { ...current, ...summary } : current)
    })
  }

  function remove(id: string) {
    return manage('deleting', async () => {
      await deleteConversation(id)
      setConversations((current) => current.filter((conversation) => conversation.id !== id))
      if (active?.id === id) activate(null)
    })
  }

  async function send(documentIds: string[] | null) {
    const question = draft.trim()
    if (!question || question.length > MAX_QUESTION_LENGTH || restoring || requestInProgress.current) return
    setDraft('')
    setError(undefined)
    setPending({ id: crypto.randomUUID(), role: 'user', content: question })

    try {
      const sent = await manage('asking', async () => {
        let conversation = active
        if (!conversation) {
          conversation = await createConversation(documentIds)
          setActive(conversation)
          rememberActiveId(conversation.id)
          updateSummary(conversation)
        }
        const result = await askQuestion(conversation.id, question, documentIds)
        updateSummary(result.conversation)
        setActive({
          ...result.conversation,
          document_ids: documentIds,
          messages: [...conversation.messages, ...result.messages],
        })
      })
      if (!sent) setDraft(question)
    } finally {
      setPending(undefined)
    }
  }

  return {
    active, conversations, messages, draft, pending, error, historyError, restoring, operation,
    busy: restoring || operation !== undefined,
    setDraft, send, create, open, rename, remove,
    retry: () => setRefresh((current) => current + 1),
  }
}
