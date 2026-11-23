export type DocumentStatus = 'processing' | 'ready' | 'deleting'

export interface IndexedDocument {
  id: string
  filename: string
  page_count: number
  chunk_count: number
  status: DocumentStatus
  size_bytes: number
  created_at: string
}

export interface FileIssue {
  filename: string
  code: string
  message: string
}

export interface UploadResult {
  documents: IndexedDocument[]
  errors: FileIssue[]
}

export interface Health {
  status: 'ok'
  groq_configured: boolean
  demo_mode?: boolean
  turnstile_site_key?: string | null
  turnstile_action?: string | null
  max_question_length?: number
}

export interface Source {
  id: number
  document_id: string
  filename: string
  page_number: number
  chunk_id: string
  chunk_ids?: string[]
  snippet: string
}

export interface ChatTurn {
  role: 'user' | 'assistant'
  content: string
}

export interface ChatAnswer {
  status: 'answered' | 'insufficient_context'
  answer: string
  sources: Source[]
}

export interface ChatResult extends ChatAnswer {
  conversation_id: string
  conversation: ConversationSummary
  messages: [ConversationMessage, ConversationMessage]
}

export interface ChatMessage extends ChatTurn {
  id: string
  sources?: Source[]
}

export interface ConversationSummary {
  id: string
  title: string
  created_at: string
  updated_at: string
}

export interface ConversationMessage extends ChatTurn {
  id: string
  conversation_id: string
  created_at: string
  source_references: Source[] | null
}

export interface Conversation extends ConversationSummary {
  document_ids: string[] | null
  messages: ConversationMessage[]
}
