import { useEffect, useRef, useState } from 'react'
import { askDemoQuestion, errorMessage, RequestError } from '../services/api'
import type { ChatMessage, Source } from '../types/api'

const DEMO_CHAT_KEY = 'askmydocs:public-demo-chat:v1'
const MAX_DEMO_MESSAGES = 40

function isSource(value: unknown): value is Source {
  if (typeof value !== 'object' || value === null) return false
  const source = value as Partial<Source>
  return typeof source.id === 'number' && typeof source.filename === 'string'
    && typeof source.document_id === 'string' && typeof source.page_number === 'number'
    && typeof source.snippet === 'string'
}

function isMessage(value: unknown): value is ChatMessage {
  if (typeof value !== 'object' || value === null) return false
  const message = value as Partial<ChatMessage>
  return typeof message.id === 'string' && typeof message.content === 'string'
    && (message.role === 'user' || message.role === 'assistant')
    && (message.sources === undefined || (Array.isArray(message.sources) && message.sources.every(isSource)))
}

function loadDemoChat(): ChatMessage[] {
  try {
    const messages: unknown = JSON.parse(sessionStorage.getItem(DEMO_CHAT_KEY) ?? '[]')
    return Array.isArray(messages) ? messages.filter(isMessage).slice(-MAX_DEMO_MESSAGES) : []
  } catch {
    return []
  }
}

function rateLimitMessage(seconds?: number): string {
  if (!seconds) return 'The demo is at its request limit. Please try again in a little while.'
  const count = Math.ceil(seconds >= 60 ? seconds / 60 : seconds)
  const unit = seconds >= 60 ? 'minute' : 'second'
  return `The demo is at its request limit. Try again in about ${count} ${unit}${count === 1 ? '' : 's'}.`
}

export function useDemoConversation(maxQuestionLength: number) {
  const [messages, setMessages] = useState(loadDemoChat)
  const [draft, setDraft] = useState('')
  const [pending, setPending] = useState<ChatMessage>()
  const [error, setError] = useState<string>()
  const [retryAt, setRetryAt] = useState<number>()
  const asking = useRef(false)

  useEffect(() => {
    try {
      if (messages.length) sessionStorage.setItem(DEMO_CHAT_KEY, JSON.stringify(messages))
      else sessionStorage.removeItem(DEMO_CHAT_KEY)
    } catch {
      // Keep the current chat in memory when browser storage is unavailable.
    }
  }, [messages])

  useEffect(() => {
    if (retryAt === undefined) return
    const timeout = window.setTimeout(() => {
      setRetryAt(undefined)
      setError(undefined)
    }, Math.min(Math.max(0, retryAt - Date.now()), 2 ** 31 - 1))
    return () => window.clearTimeout(timeout)
  }, [retryAt])

  async function send(documentIds: string[] | null, token?: string) {
    const question = draft.trim()
    if (!question || question.length > maxQuestionLength || asking.current || retryAt !== undefined) return
    asking.current = true
    const user: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: question }
    setPending(user)
    setDraft('')
    setError(undefined)

    try {
      const history = messages.slice(-8).map(({ role, content }) => ({ role, content: content.slice(0, 4_000) }))
      const result = await askDemoQuestion(question, documentIds, history, token)
      const assistant: ChatMessage = {
        id: crypto.randomUUID(), role: 'assistant', content: result.answer, sources: result.sources,
      }
      setMessages((current) => [...current, user, assistant].slice(-MAX_DEMO_MESSAGES))
    } catch (failure) {
      setDraft(question)
      if (failure instanceof RequestError && failure.status === 429) {
        setError(rateLimitMessage(failure.retryAfterSeconds))
        if (failure.retryAfterSeconds) setRetryAt(Date.now() + failure.retryAfterSeconds * 1_000)
      } else if (failure instanceof RequestError && failure.code === 'demo_verification_required') {
        setError('Verify your browser again, then send your question.')
      } else if (failure instanceof RequestError && failure.status === 503) {
        setError('The demo is temporarily unavailable. The service may be waking up. Try your question again in a moment.')
      } else {
        setError(errorMessage(failure))
      }
    } finally {
      asking.current = false
      setPending(undefined)
    }
  }

  function clear(): boolean {
    if (asking.current) return false
    setMessages([])
    setDraft('')
    if (retryAt === undefined) setError(undefined)
    return true
  }

  return { messages, draft, pending, error, busy: pending !== undefined, rateLimited: retryAt !== undefined, setDraft, send, clear }
}
