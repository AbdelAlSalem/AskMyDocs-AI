import type { ReactNode, RefObject } from 'react'
import { ArrowUp, CircleAlert, Layers, LoaderCircle } from 'lucide-react'
import { MAX_QUESTION_LENGTH } from '../hooks/useConversation'

interface ComposerProps {
  input: RefObject<HTMLTextAreaElement | null>
  draft: string
  onDraftChange: (draft: string) => void
  onSend: () => void
  pending: boolean
  disabled: boolean
  maxQuestionLength?: number
  verification?: ReactNode
  scope: string
  canSearch: boolean
  error?: string
}

export function Composer({ input, draft, onDraftChange, onSend, pending, disabled, maxQuestionLength = MAX_QUESTION_LENGTH, verification, scope, canSearch, error }: ComposerProps) {
  const canSend = canSearch && !disabled && draft.trim().length > 0 && draft.trim().length <= maxQuestionLength

  return (
    <footer className="composer-region">
      <div className="composer-width">
        {error && (
          <div className="chat-error" role="alert">
            <CircleAlert size={18} aria-hidden="true" />
            <div><p>Could not generate an answer</p><span>{error}</span></div>
          </div>
        )}
        {verification}
        <form className="composer" onSubmit={(event) => { event.preventDefault(); if (canSend) onSend() }}>
          <label htmlFor="document-question" className="sr-only">Ask about your documents</label>
          <textarea
            id="document-question"
            ref={input}
            rows={2}
            dir="auto"
            placeholder="Ask a question about your documents…"
            aria-describedby="document-scope"
            value={draft}
            maxLength={maxQuestionLength}
            disabled={disabled}
            onChange={(event) => onDraftChange(event.currentTarget.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault()
                if (canSend) onSend()
              }
            }}
          />
          <div className="composer-toolbar">
            <span id="document-scope" className="composer-scope"><Layers size={14} aria-hidden="true" /> {scope}</span>
            <button className="send-button" type="submit" aria-label="Send question" title="Send question" disabled={!canSend}>
              {pending ? <LoaderCircle size={19} className="spin" /> : <ArrowUp size={20} strokeWidth={2.3} />}
            </button>
          </div>
        </form>
        <p className="composer-note">Check the source pages before relying on an answer.</p>
      </div>
    </footer>
  )
}
