import type { ChatAnswer, ChatResult, ChatTurn, Conversation, ConversationSummary, Health, IndexedDocument, UploadResult } from '../types/api'

const apiUrl = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '')

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

export class RequestError extends Error {
  constructor(message: string, readonly status: number, readonly code?: string, readonly retryAfterSeconds?: number) {
    super(message)
  }
}

function requestError(response: Response, body: unknown): RequestError {
  const detail = isRecord(body) ? body.detail : undefined
  const message = isRecord(detail) && typeof detail.message === 'string' ? detail.message
    : response.status === 422 ? 'Check the request and try again.'
      : `The backend could not complete the request (HTTP ${response.status}).`
  const code = isRecord(detail) && typeof detail.code === 'string' ? detail.code : undefined
  const retryAfter = Number(response.headers.get('Retry-After'))
  return new RequestError(message, response.status, code, Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter : undefined)
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(`${apiUrl}/api${path}`, init)
  } catch {
    throw new Error('Could not reach the backend. Check that FastAPI is running and try again.')
  }
}

async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await request(path, init)
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) throw requestError(response, body)
  if (body === null) throw new Error('The backend returned an unreadable response.')
  return body as T
}

export async function getHealth(signal?: AbortSignal): Promise<Health> {
  const result = await getJson<Health>('/health', { signal })
  if (result.status !== 'ok' || typeof result.groq_configured !== 'boolean') {
    throw new Error('The backend returned an unrecognized health response.')
  }
  return result
}

export async function listDocuments(signal?: AbortSignal): Promise<IndexedDocument[]> {
  const result = await getJson<{ documents: IndexedDocument[] }>('/documents', { signal })
  return result.documents
}

export async function uploadDocuments(files: File[]): Promise<UploadResult> {
  const data = new FormData()
  for (const file of files) data.append('files', file)

  const response = await request('/documents/upload', { method: 'POST', body: data })
  const body: unknown = await response.json().catch(() => null)
  const isUploadResult = isRecord(body) && Array.isArray(body.documents) && Array.isArray(body.errors)
  if (isUploadResult && (response.ok || [400, 409, 413, 415, 422, 503].includes(response.status))) {
    return body as unknown as UploadResult
  }
  throw requestError(response, body)
}

export async function deleteDocument(id: string): Promise<void> {
  const response = await request(`/documents/${encodeURIComponent(id)}`, { method: 'DELETE' })
  if (!response.ok) throw requestError(response, await response.json().catch(() => null))
}

export async function listConversations(signal?: AbortSignal): Promise<ConversationSummary[]> {
  const result = await getJson<{ conversations: ConversationSummary[] }>('/conversations', { signal })
  return result.conversations
}

export function getConversation(id: string, signal?: AbortSignal): Promise<Conversation> {
  return getJson<Conversation>(`/conversations/${encodeURIComponent(id)}`, { signal })
}

export function createConversation(documentIds: string[] | null): Promise<Conversation> {
  return getJson<Conversation>('/conversations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_ids: documentIds }),
  })
}

export function renameConversation(id: string, title: string): Promise<ConversationSummary> {
  return getJson<ConversationSummary>(`/conversations/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  })
}

export async function deleteConversation(id: string): Promise<void> {
  const response = await request(`/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' })
  if (!response.ok) throw requestError(response, await response.json().catch(() => null))
}

export function askQuestion(conversationId: string, question: string, documentIds: string[] | null): Promise<ChatResult> {
  return getJson<ChatResult>('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ conversation_id: conversationId, question, document_ids: documentIds }),
  })
}

export function askDemoQuestion(question: string, documentIds: string[] | null, history: ChatTurn[], token?: string): Promise<ChatAnswer> {
  return getJson<ChatAnswer>('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, document_ids: documentIds, history, turnstile_token: token }),
  })
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Something went wrong. Try again.'
}
