# Learn RAG with AskMyDocs AI

Start with the sample PDF and ask, “What is the launch date for Project Orbit?” Follow the answer back to page 1. Then ask “أين ستقام الورشة التجريبية؟” and check page 2. These two queries take the same path through the application.

## Vocabulary

| English | Arabic | Meaning in this app |
| --- | --- | --- |
| Retrieval-augmented generation (RAG) | التوليد المعزَّز بالاسترجاع | Search for evidence before asking a model to write |
| Ingestion | استيعاب البيانات | Turn an uploaded PDF into searchable passages |
| Chunk | مقطع نصي | A short piece of text from one page |
| Embedding | تمثيل متجهي | An array of numbers produced from text |
| Vector database | قاعدة بيانات متجهية | An index that can find nearby vectors |
| Similarity | التشابه | How close the question and passage vectors are |
| Retrieval | الاسترجاع | Select candidate evidence from the index |
| Generation | التوليد | Write an answer from the question and evidence |
| Citation | إحالة إلى المصدر | A numbered reference to a filename and page |

## 1. Trace an upload

1. Open [api/routes.py](backend/app/api/routes.py) and find `upload_documents`. Its input is a multipart batch. It calls the document service once per file and reports both successes and failures.
2. Read [services/pdf.py](backend/app/services/pdf.py). `read_pdf_upload` checks the extension, byte limit, and `%PDF-` header. `extract_pages` returns `PDFPage` records with one-based page numbers. Open the sample and compare the extracted phrase with the page heading.
3. Read [rag/chunking.py](backend/app/rag/chunking.py). `split_pages` sends one page at a time to LangChain’s recursive splitter. If one page contains two chunks, both point back to the same page. Overlap can keep a sentence near a chunk boundary visible to both chunks.
4. Read [services/documents.py](backend/app/services/documents.py) and [services/catalog.py](backend/app/services/catalog.py). The service hashes the uploaded bytes to identify duplicates, creates a generated ID, writes the PDF, and marks it ready after inserting its vectors. Processing and deletion statuses let startup clean up operations that were interrupted.

## 2. Trace retrieval

Open [rag/embeddings.py](backend/app/rag/embeddings.py). `LocalEmbeddings` creates the embedder when needed. Text with related meaning can have similar vectors, even if the question uses different words. For example, “When does Orbit start?” may retrieve a passage that says “The launch date is 18 June 2027.” The model was trained to compare many languages, so it can also match an Arabic question to text with a similar meaning.

Open [rag/vector_store.py](backend/app/rag/vector_store.py). `VectorStore.add` stores each vector and its source metadata. `VectorStore.retrieve` filters by document ID and combines cosine similarity with exact words and phrases, which helps with values like VLAN IDs. Track how the page number follows `Chunk` → Chroma metadata → `RetrievedChunk`.

Try changing `RAG_TOP_K` from 4 to 1 in `backend/.env`, then restart the API. Ask a question about two facts on different pages and count the source cards. A smaller value reduces the text sent to the provider and can omit evidence. A larger value can include more evidence or bring in distracting passages.

## 3. Trace answer generation

Open [rag/answer.py](backend/app/rag/answer.py). `AnswerService.answer` retrieves passages and numbers them. If nothing qualifies, it returns an insufficient-context message immediately. Otherwise, it calls the generator, verifies quotes and numbers, and assigns one citation per source page. For a sum or difference, the verifier checks the equation against operands quoted from the supporting passages.

Open [rag/prompt.py](backend/app/rag/prompt.py). `build_prompt` sends three distinct fields to the model.

- `question` is what the user wants to know.
- `conversation_history` gives context to phrases such as “that workshop.”
- `retrieved_passages` carries the evidence, with a number, filename, and page for each passage.

For the English sample, the generator can return the claim “Project Orbit launches on **18 June 2027**” with an exact quote from the launch brief. The backend verifies the quote and adds `[1]` with the original filename and page. Inspect the cited text to decide whether it supports the claim.

Open [services/groq_client.py](backend/app/services/groq_client.py). `GroqGenerator.generate` sends the request using the official SDK, requires a structured answer, and maps exceptions to messages the UI can display. A missing key becomes a configuration error only when the answer needs a provider call.

## 4. Trace the browser state

- [pages/Workspace.tsx](frontend/src/pages/Workspace.tsx) connects the document sidebar and chat.
- [hooks/useWorkspace.ts](frontend/src/hooks/useWorkspace.ts) tracks files, selected IDs, and upload/delete state.
- [hooks/useConversation.ts](frontend/src/hooks/useConversation.ts) opens saved conversations and sends the active conversation ID with each question. The API reads recent messages from SQLite and saves each successful turn with its verified citations.
- [services/api.ts](frontend/src/services/api.ts) is the boundary between the UI and the API. It handles JSON and multipart requests.
- [components/ChatMessages.tsx](frontend/src/components/ChatMessages.tsx) renders answer Markdown and source cards. Direction follows the text, so Arabic messages can flow right to left.

## 5. Read a test and try a change

Start with [test_documents.py](backend/tests/test_documents.py). The tests inject a small deterministic embedder while using real Chroma storage. A test can then ask whether a filename and page survived indexing without waiting for a model download.

Next read [test_api.py](backend/tests/test_api.py) and [test_groq_client.py](backend/tests/test_groq_client.py). The API tests use FastAPI's TestClient. SDK tests intercept outbound requests and can make the provider return a rate limit or timeout deliberately.

Upload the sample, inspect its pages, and write down three questions with evidence and three whose answers are absent. Include a question that asks for two facts across pages. Try changing only the similarity threshold and record the returned sources with each question. This gives you a small evaluation set for reasoning about retrieval quality before changing the prompt.
