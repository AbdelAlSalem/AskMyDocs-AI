import { expect, test } from '@playwright/test'
import type { ChatResult, Conversation, IndexedDocument } from '../src/types/api'

const timestamp = '2026-01-01T00:00:00Z'
const scope = ['selected']
const conversation: Conversation = {
  id: 'restored-conversation',
  title: 'An active conversation',
  document_ids: null,
  created_at: timestamp,
  updated_at: timestamp,
  messages: [],
}
const documents: IndexedDocument[] = ['selected', 'excluded'].map((id) => ({
  id,
  filename: `${id}.pdf`,
  page_count: 2,
  chunk_count: 2,
  size_bytes: 1200,
  status: 'ready',
  created_at: timestamp,
}))
const datesQuestion = 'Are any dates or deadlines mentioned?'

for (const demo of [false, true]) {
  test(`a suggestion submits like typing with the same scope in ${demo ? 'demo' : 'local'} mode`, async ({ page }) => {
    const question = demo ? 'ما أهم النقاط في المستند؟' : 'Summarize the main points.'
    const answer = demo ? 'ملخص مدعوم بالمستند.' : 'A summary grounded in the selected PDF.'
    const result: ChatResult = {
      status: 'answered',
      conversation_id: conversation.id,
      conversation,
      answer,
      sources: [],
      messages: [
        { id: 'question', conversation_id: conversation.id, role: 'user', content: question, created_at: timestamp, source_references: null },
        { id: 'answer', conversation_id: conversation.id, role: 'assistant', content: answer, created_at: timestamp, source_references: [] },
      ],
    }
    const chatRequests: unknown[] = []
    let gate = Promise.withResolvers<void>()
    let fail = false
    await page.route('**/api/**', async (route) => {
      switch (new URL(route.request().url()).pathname) {
        case '/api/health':
          return route.fulfill({ json: { status: 'ok', groq_configured: true, demo_mode: demo } })
        case '/api/documents':
          return route.fulfill({ json: { documents } })
        case '/api/conversations':
          if (!demo && route.request().method() === 'GET') return route.fulfill({ json: { conversations: [conversation] } })
          break
        case `/api/conversations/${conversation.id}`:
          if (!demo) return route.fulfill({ json: conversation })
          break
        case '/api/chat':
          if (route.request().method() !== 'POST') break
          chatRequests.push(route.request().postDataJSON())
          await gate.promise
          if (fail) return route.fulfill({ status: 500, json: { detail: { message: 'Synthetic backend failure' } } })
          return route.fulfill({ json: demo ? { status: result.status, answer, sources: [] } : result })
      }
      return route.abort()
    })

    const input = page.getByRole('textbox', { name: 'Ask about your documents' })
    const send = page.getByRole('button', { name: 'Send question' })
    const suggestion = page.getByRole('button', { name: question, exact: true })
    async function openWelcome() {
      await page.goto('/')
      await expect(suggestion).toBeVisible()
      await page.getByRole('checkbox', { name: 'Select selected.pdf for search' }).check()
      await expect(page.getByText('1 selected document', { exact: true })).toBeVisible()
      if (!demo) await expect(page.getByText(conversation.title, { exact: true }).first()).toBeVisible()
    }

    await openWelcome()
    await suggestion.click()
    await expect(input).toBeFocused()
    await expect(input).toHaveValue(question)
    expect(chatRequests).toHaveLength(0)
    await send.click()
    await expect(page.getByText('Searching documents…')).toBeVisible()
    await expect(send).toBeDisabled()
    gate.resolve()
    await expect(page.getByRole('article', { name: 'AskMyDocs AI answer' })).toContainText(answer)

    if (demo) await page.getByRole('button', { name: 'New chat' }).click()
    await openWelcome()
    gate = Promise.withResolvers<void>()
    await input.fill(question)
    await input.press('Enter')
    await expect(page.getByText('Searching documents…')).toBeVisible()
    gate.resolve()
    await expect(page.getByRole('article', { name: 'AskMyDocs AI answer' })).toContainText(answer)
    const payload = { question, document_ids: scope }
    const expected = demo ? { ...payload, history: [] } : { ...payload, conversation_id: conversation.id }
    expect(chatRequests).toEqual([expected, expected])

    if (demo) await page.getByRole('button', { name: 'New chat' }).click()
    await openWelcome()
    fail = true
    await page.getByRole('button', { name: datesQuestion, exact: true }).click()
    await send.click()
    await expect(page.getByRole('alert')).toContainText('Synthetic backend failure')
    await expect(input).toHaveValue(datesQuestion)
    await expect(send).toBeEnabled()
  })
}
