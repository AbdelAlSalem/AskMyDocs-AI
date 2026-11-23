import { ArrowUpRight, CalendarDays, ChevronDown, FileText, ListChecks, MessageCircle, Sparkles, UserRound } from 'lucide-react'
import { memo } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ChatMessage, Source } from '../types/api'

const suggestions = [
  { text: 'Summarize the main points.', icon: ListChecks, language: 'en' },
  { text: 'Are any dates or deadlines mentioned?', icon: CalendarDays, language: 'en' },
  { text: 'ما أهم النقاط في المستند؟', icon: MessageCircle, language: 'ar' },
]

export function ChatWelcome({ onSuggestion, demo }: { onSuggestion: (question: string) => void; demo?: boolean }) {
  return (
    <div className="chat-welcome">
      <div className="hero-mark" aria-hidden="true">
        <span className="hero-sheet hero-sheet-left" />
        <span className="hero-sheet hero-sheet-right" />
        <span className="hero-sheet hero-sheet-front"><FileText size={32} strokeWidth={1.55} /></span>
        <Sparkles className="hero-sparkles" size={24} strokeWidth={1.65} />
      </div>
      <h1>Ask your <span>documents.</span></h1>
      <p>Ask questions about {demo ? 'the sample PDFs' : 'your PDFs'} in English or Arabic and find the pages that support each answer.</p>
      <div className="suggestions" aria-label="Example questions">
        {suggestions.map(({ text, icon: Icon, language }) => (
          <button className="suggestion" type="button" key={text} onClick={() => onSuggestion(text)}>
            <span className="suggestion-icon"><Icon size={16} aria-hidden="true" /></span>
            <span className="suggestion-text" lang={language} dir="auto">{text}</span>
            <ArrowUpRight className="suggestion-arrow" size={14} aria-hidden="true" />
          </button>
        ))}
      </div>
    </div>
  )
}

function Sources({ sources, knownDocumentIds }: { sources: Source[]; knownDocumentIds?: ReadonlySet<string> }) {
  return (
    <section className="sources" aria-label="Retrieved sources">
      <h3>Retrieved sources <span>{sources.length}</span></h3>
      <div className="source-grid">
        {sources.map((source) => (
          <details className="source-card" key={source.id}>
            <summary>
              <span className="source-number" aria-label={`Source ${source.id}`}>{source.id}</span>
              <span className="source-description">
                <bdi title={source.filename}>{source.filename}</bdi>
                <span>Page {source.page_number}</span>
                {knownDocumentIds && !knownDocumentIds.has(source.document_id) && (
                  <span className="source-historical">Historical source · PDF deleted</span>
                )}
              </span>
              <ChevronDown size={15} className="source-chevron" aria-hidden="true" />
            </summary>
            <p dir="auto">{source.snippet}</p>
          </details>
        ))}
      </div>
    </section>
  )
}

export const ChatBubble = memo(function ChatBubble({ message, knownDocumentIds }: { message: ChatMessage; knownDocumentIds?: ReadonlySet<string> }) {
  const isUser = message.role === 'user'

  return (
    <article className={`chat-message ${isUser ? 'chat-message-user' : 'chat-message-assistant'}`} aria-label={isUser ? 'Your message' : 'AskMyDocs AI answer'}>
      <span className="message-avatar" aria-hidden="true">{isUser ? <UserRound size={17} /> : <Sparkles size={17} />}</span>
      <div className="message-content">
        <p className="message-author">{isUser ? 'You' : 'AskMyDocs AI'}</p>
        {isUser ? (
          <p className="user-bubble" dir="auto">{message.content}</p>
        ) : (
          <div className="assistant-bubble markdown" dir="auto">
            <Markdown
              remarkPlugins={[remarkGfm]}
              skipHtml
              disallowedElements={['img', 'input']}
              components={{ a: ({ children }) => <span>{children}</span> }}
            >
              {message.content}
            </Markdown>
          </div>
        )}
        {!!message.sources?.length && <Sources sources={message.sources} knownDocumentIds={knownDocumentIds} />}
      </div>
    </article>
  )
})

export function ChatLoading() {
  return (
    <div className="chat-message chat-message-assistant" role="status">
      <span className="message-avatar" aria-hidden="true"><Sparkles size={17} /></span>
      <div className="message-content">
        <p className="message-author">AskMyDocs AI</p>
        <p className="assistant-bubble loading-bubble">
          <span className="loading-dots" aria-hidden="true"><i /><i /><i /></span>
          Searching documents…
        </p>
      </div>
    </div>
  )
}
