import { useCallback, useEffect, useRef, useState } from 'react'
import { deleteDocument, errorMessage, getHealth, listDocuments, uploadDocuments } from '../services/api'
import type { FileIssue, Health, IndexedDocument } from '../types/api'

type Mutation = { type: 'upload'; count: number } | { type: 'delete'; id: string }

export function useWorkspace() {
  const [documents, setDocuments] = useState<IndexedDocument[]>([])
  const [health, setHealth] = useState<Health | null>()
  const [backendConfig, setBackendConfig] = useState<Health>()
  const [checkingHealth, setCheckingHealth] = useState(true)
  const [loadingDocuments, setLoadingDocuments] = useState(true)
  const [loadError, setLoadError] = useState<string>()
  const [refresh, setRefresh] = useState(0)
  const [mutation, setMutation] = useState<Mutation>()
  const [actionError, setActionError] = useState<string>()
  const [issues, setIssues] = useState<FileIssue[]>([])
  const [statusMessage, setStatusMessage] = useState('')
  const [selection, setSelection] = useState<Set<string> | null>(null)
  const revision = useRef(0)
  const mutating = useRef(false)

  useEffect(() => {
    const controller = new AbortController()
    const signal = controller.signal
    const listRevision = revision.current
    setCheckingHealth(true)
    setLoadingDocuments(true)
    setLoadError(undefined)

    void getHealth(signal).then(
      (result) => {
        if (signal.aborted) return
        setHealth(result)
        setBackendConfig(result)
      },
      () => { if (!signal.aborted) setHealth(null) },
    ).finally(() => { if (!signal.aborted) setCheckingHealth(false) })
    void listDocuments(signal).then(
      (result) => {
        if (!signal.aborted && revision.current === listRevision) setDocuments(result)
      },
      (error: unknown) => { if (!signal.aborted) setLoadError(errorMessage(error)) },
    ).finally(() => { if (!signal.aborted) setLoadingDocuments(false) })

    return () => controller.abort()
  }, [refresh])

  const readyDocuments = documents.filter((document) => document.status === 'ready')
  const documentIds = selection === null
    ? null
    : readyDocuments.filter((document) => selection.has(document.id)).map((document) => document.id)

  const restoreScope = useCallback((ids: string[] | null) => {
    setSelection(ids === null ? null : new Set(ids))
  }, [])

  function toggleDocument(id: string) {
    setSelection((current) => {
      const next = new Set(readyDocuments.filter((document) => current?.has(document.id)).map((document) => document.id))
      if (next.has(id)) next.delete(id)
      else if (next.size < 100) next.add(id)
      return next
    })
  }

  function dismissNotices() {
    setActionError(undefined)
    setIssues([])
  }

  async function upload(files: File[]) {
    if (!files.length || mutating.current) return
    mutating.current = true
    dismissNotices()
    setStatusMessage('')
    setMutation({ type: 'upload', count: files.length })

    try {
      const result = await uploadDocuments(files)
      if (result.documents.length > 0) {
        revision.current += 1
        setDocuments((current) => {
          const updatedIds = new Set(result.documents.map((document) => document.id))
          return [...result.documents, ...current.filter((document) => !updatedIds.has(document.id))]
        })
        setLoadError(undefined)
        setStatusMessage(`${result.documents.length} ${result.documents.length === 1 ? 'PDF' : 'PDFs'} indexed.`)
      }
      setIssues(result.errors)
    } catch (error) {
      setActionError(errorMessage(error))
    } finally {
      mutating.current = false
      setMutation(undefined)
    }
  }

  async function remove(document: IndexedDocument) {
    if (mutating.current) return
    mutating.current = true
    dismissNotices()
    setStatusMessage('')
    setMutation({ type: 'delete', id: document.id })
    try {
      await deleteDocument(document.id)
      revision.current += 1
      setDocuments((current) => current.filter((item) => item.id !== document.id))
      setStatusMessage(`${document.filename} deleted.`)
    } catch (error) {
      setActionError(errorMessage(error))
      setRefresh((current) => current + 1)
    } finally {
      mutating.current = false
      setMutation(undefined)
    }
  }

  return {
    documents, readyCount: readyDocuments.length, health, backendConfig, checkingHealth, loadingDocuments, loadError,
    mutation, actionError, issues, statusMessage, documentIds,
    toggleDocument, upload, remove, dismissNotices, restoreScope,
    selectAll: () => setSelection(null),
    retry: () => setRefresh((current) => current + 1),
  }
}
