# AskMyDocs AI on free plans

This guide prepares a personal, non-commercial portfolio demo. Create resources or publish the repository only after approving those actions separately. Use free accounts and provider subdomains. Leave payment methods, paid add-ons, and upgrades disabled. If a provider requires a payment method for your account, stop at that screen.

## Architecture

| Component | Plan | Public behavior |
| --- | --- | --- |
| Source | GitHub Free | Version the code and approved synthetic corpus |
| Frontend | Vercel Hobby | Serve the Vite build as static files at a `.vercel.app` address |
| API | Render Free web service | One Uvicorn worker on a `.onrender.com` address |
| Generation | GroqCloud Free | Generate grounded responses from synthetic excerpts |
| Bot protection | Cloudflare Turnstile Free | Optionally require a verified challenge for each chat request |

[GitHub plans](https://docs.github.com/en/get-started/learning-about-github/githubs-plans) · [Vercel Hobby](https://vercel.com/docs/plans/hobby) · [Render compute plans](https://render.com/docs/compute-plans) · [Groq billing](https://console.groq.com/docs/billing-faqs) · [Turnstile plans](https://developers.cloudflare.com/turnstile/plans/)

With `DEMO_MODE=true`, FastAPI reads the checked-in corpus generated from the synthetic [Project Orbit PDF](samples/project-orbit.pdf). The corpus vectors are precomputed as normalized `float32` embeddings with the full local `paraphrase-multilingual-MiniLM-L12-v2` model. The public API embeds questions with the official AVX2 uint8 ONNX export and native SentencePiece tokenizer from revision `e8f8c21`. It scans the in-memory vectors using exact cosine similarity and applies the local lexical scoring for terms and identifiers. It can retrieve both pages when a question combines two facts. The shared grounded answer and citation verifier checks Groq's output against those passages. Missing model files or stale corpus vectors prevent startup. [Pinned upstream ONNX exports](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/e8f8c211226b894fcb81acc59f3b34ba3efd5f42/onnx)

The public API serves the sample list and chat. It blocks uploads, document deletion, and all conversation-storage routes. Each tab has its own browser session transcript. Recent messages from that tab may be sent to Groq to interpret follow-ups. Retrieval still uses the current question, and retrieved PDF passages supply the evidence. Avoid typing personal or confidential information into the public demo.

Summary and date questions scan only the selected sample passages. They use the same context budget and at most one Groq request per chat. Large summaries reduce page ranges to bounded excerpts before generation. The demo continues to use its existing semantic pipeline for specific questions.

Local mode indexes uploaded PDFs using Sentence Transformers, PyTorch, and ChromaDB. SQLite stores its document metadata and saved conversations.

## Windows preparation

Run from the project root. Use 64-bit Python 3.12 on an x86-64 CPU with AVX2 and the Node.js version required by [Vite](https://vite.dev/guide/).

```powershell
py -3.12 --version
node --version
py -3.12 -m venv .venv-demo
& .\.venv-demo\Scripts\python.exe -m pip install -r .\backend\requirements-demo.txt
& .\.venv-demo\Scripts\python.exe .\backend\scripts\prepare_demo_model.py
npm.cmd --prefix .\frontend ci
npm.cmd --prefix .\frontend run build
```

The model preparation script fetches the pinned encoder and tokenizer into `backend/app/demo_model/` and verifies their SHA-256 checksums. Use `--check` to verify files that are already present.

If the approved sample changes, regenerate the corpus and its full-precision embeddings using the existing local virtual environment. In a fresh workspace, complete the [local setup](README.md) first.

```powershell
& .\.venv\Scripts\python.exe .\backend\scripts\build_demo_corpus.py
& .\.venv\Scripts\python.exe .\backend\scripts\build_demo_embeddings.py
```

Verify the corpus, vectors, and prepared model files before running the backend tests.

```powershell
& .\.venv\Scripts\python.exe .\backend\scripts\build_demo_corpus.py --check
& .\.venv\Scripts\python.exe .\backend\scripts\build_demo_embeddings.py --check
& .\.venv-demo\Scripts\python.exe .\backend\scripts\prepare_demo_model.py --check
& .\.venv\Scripts\python.exe -m pytest .\backend
```

To run the demo API without sending requests to Groq, use a new PowerShell terminal. The single space overrides any key in your local backend environment file for this terminal.

```powershell
$env:DEMO_MODE = 'true'
$env:GROQ_API_KEY = ' '
& .\.venv-demo\Scripts\python.exe -m uvicorn app.main:app --app-dir .\backend --host 127.0.0.1 --port 8010 --no-proxy-headers
```

In a second terminal, confirm `demo_mode` is `true`, `groq_configured` is `false`, and the document list contains only the sample.

```powershell
Invoke-RestMethod 'http://127.0.0.1:8010/api/health'
Invoke-RestMethod 'http://127.0.0.1:8010/api/documents'
$env:VITE_API_URL = 'http://127.0.0.1:8010'
npm.cmd --prefix .\frontend run dev
```

Stop any Vite process already using port 5173 before starting the demo frontend. Open [the local demo](http://127.0.0.1:5173). The page should show the demo status and a read-only sample list. For live local questions, stop the demo API, set `GROQ_API_KEY` in its terminal to a key you have confirmed belongs to a Free project, and restart it. Keep the key out of the command history and all frontend environment variables.

Use these questions to verify the sample after enabling a Free key.

| Question | Expected evidence |
| --- | --- |
| What is the launch date for Project Orbit? | 18 June 2027, page 1 |
| أين ستقام الورشة التجريبية؟ | مكتبة النور في عمّان، page 2 |
| When does the project launch, and where is the workshop? | Both facts, pages 1 and 2 |
| What is the catering budget? | Insufficient context |

## GitHub preparation

Inspect the working tree and the ignore rules. Review the complete file list before any future commit or push.

```powershell
git status --short --untracked-files=all
git check-ignore -v backend/.env frontend/.env data/documents.sqlite3 data/conversations.sqlite3 data/chroma/chroma.sqlite3 backend/app/demo_model/
git ls-files -ci --exclude-standard
git ls-files --cached --others --exclude-standard -- '*.pdf' 'data/**' 'samples/**' 'backend/app/demo_data/**'
```

The tracked-and-ignored check should print nothing. The last command should list exactly four files: `data/.gitkeep`, `samples/project-orbit.pdf`, and `corpus.json` and `embeddings.json` under `backend/app/demo_data/`. Include the application source, tests, requirements, frontend lockfile, blank-key environment examples, documentation and reviewed synthetic screenshot. Keep local diagnostic reports, uploaded files, databases, downloaded model weights, and credentials out of the release. If a credential ever entered a commit, revoke or rotate it and review the affected history before publishing. Publish only the reviewed branch and release tag. [GitHub Free plans](https://docs.github.com/en/get-started/learning-about-github/githubs-plans)

## Groq Free setup

1. Open GroqCloud and confirm the project belongs to an organization on the **Free** plan. Keep it on Free without a payment method.
2. Create a dedicated project key for the demo when authorized. Add it only to the Render service as the secret `GROQ_API_KEY`. Use `openai/gpt-oss-20b`.
3. Check the model's request and token limits in the Free organization's limits dashboard. The SDK uses the standard inference endpoint with automatic retries disabled.

The API key does not encode the billing plan in a way this app checks. The operator must verify Free in the provider dashboard. An app limiter does not turn a key from a billed account into a free key. [Groq account limits](https://console.groq.com/docs/rate-limits) · [Groq billing FAQs](https://console.groq.com/docs/billing-faqs)

## Render Free API deployment

After approval, use a free workspace with **no payment method**. Choose **New → Web Service**, connect the approved repository, and use these settings:

| Field | Value |
| --- | --- |
| Runtime | Python |
| Root Directory | `backend` |
| Instance plan | **Free** |
| Build Command | `pip install -r requirements-demo.txt && python scripts/prepare_demo_model.py` |
| Start Command | `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --no-proxy-headers --limit-concurrency 24` |
| Health Check Path | `/api/health` |
| Auto-Deploy | Off |

Set `PYTHON_VERSION=3.12.10`, `DEMO_MODE=true`, `DEMO_ALLOWED_ORIGINS=https://<your-project>.vercel.app`, and the Free Groq settings. On Render's public ingress, set `DEMO_TRUST_RENDER_CLIENT_IP=true` so the limiter uses the header Render's Cloudflare edge rewrites. Keep the start command's `--no-proxy-headers` and leave this setting off on other hosts unless their ingress provides the same guarantee. Choose Free again on the final review screen. Keep persistent disks, datastores, and paid build options disabled. [Render web services](https://render.com/docs/web-services) · [Render free limits](https://render.com/docs/free) · [Render client-IP guidance](https://render.com/articles/host-pocketbase-on-render)

The build includes the prepared encoder and tokenizer in the deployed application. Startup requires valid model assets and checked-in corpus vectors. Record the service's generated `https://<service>.onrender.com` origin. Request `/api/health` and check `demo_mode: true` before linking the frontend. Verify the sample list, the example questions, and HTTP 403 responses for upload, PDF deletion, and conversation routes.

## Vercel Hobby frontend deployment

1. Use a GitHub repository owned by your **personal account** for the Hobby Git integration. Import it into a **Vercel Hobby** project for this non-commercial demo. Select Vite and set **Root Directory** to `frontend`. [Vercel Git ownership limits](https://vercel.com/docs/limits#connecting-a-project-to-a-git-repository)
2. Use `npm ci` for install, `npm run build` for build, and `dist` for the output directory. Keep the build static.
3. Set `VITE_API_URL=https://<service>.onrender.com` for Production. In Render, set `DEMO_ALLOWED_ORIGINS` to the exact production frontend origin, with no trailing slash or wildcard. Use the generated domains.
4. When authorized, deploy the production frontend. Copy the assigned production origin into Render and apply the backend environment update if the name differs. Redeploy the frontend after any change to `VITE_API_URL` because Vite embeds it during the build. Verify health and a sample question from the production URL. Keep Pro trials and paid options disabled. Preview URLs need separate explicit origins if you later use them.

[Vite on Vercel](https://vercel.com/docs/frameworks/frontend/vite) · [Vercel build settings](https://vercel.com/docs/builds/configure-a-build) · [Hobby plan terms](https://vercel.com/docs/plans/hobby)

### Optional Turnstile Free

Create a Free Managed widget for the exact frontend hostname, such as `<your-project>.vercel.app`. Keep the widget's site key and secret in Render. Set `DEMO_TURNSTILE_HOSTNAME` to the frontend hostname without the scheme and keep the action `demo_chat`. The API sends the public site key to React and verifies each token server-side against the hostname and action. A fresh token is needed after each send. The widget can run on the generated Vercel domain. [Turnstile setup](https://developers.cloudflare.com/turnstile/get-started/) · [Server-side token validation](https://developers.cloudflare.com/turnstile/get-started/server-side-validation/)

## Environment variables

| Variable | Default or deployment value |
| --- | --- |
| `DEMO_MODE` | `false` locally. Set `true` for the public backend |
| `GROQ_API_KEY` | Groq Free project key, backend secret |
| `GROQ_MODEL` | `openai/gpt-oss-20b` |
| `GROQ_TIMEOUT_SECONDS` | 35 seconds |
| `DEMO_ALLOWED_ORIGINS` | Comma-separated exact origins. Defaults to local Vite origins |
| `DEMO_MAX_BODY_BYTES` | 16,384 bytes |
| `DEMO_MAX_QUESTION_CHARS` | 600 characters |
| `DEMO_TOP_K` / `DEMO_MAX_CONTEXT_CHARS` | Up to 4 whole passages within 4,500 characters |
| `DEMO_MAX_HISTORY_MESSAGES` / `DEMO_MAX_HISTORY_CHARS` | Up to 4 recent messages within 1,800 characters |
| `DEMO_MAX_COMPLETION_TOKENS` | 1,536 tokens, including provider reasoning |
| `DEMO_RATE_WINDOW_SECONDS` | 60 seconds |
| `DEMO_IP_RATE_LIMIT` / `DEMO_GLOBAL_RATE_LIMIT` | 5 per IP and 15 total per window |
| `DEMO_DAILY_REQUEST_LIMIT` | 80 admitted chats per rolling 24 hours |
| `DEMO_MAX_CONCURRENT_REQUESTS` | 3 |
| `DEMO_TRUST_RENDER_CLIENT_IP` | `false`. Enable only on the Render public ingress described above |
| `DEMO_TURNSTILE_SITE_KEY` / `DEMO_TURNSTILE_SECRET_KEY` | Optional pair. The secret belongs only in the backend |
| `DEMO_TURNSTILE_HOSTNAME` / `DEMO_TURNSTILE_ACTION` | Exact widget hostname and `demo_chat` by default |
| `VITE_API_URL` | Public HTTPS API origin for the frontend build |

## Retrieval comparison

Run the same sample questions against the local and public retrieval engines. The comparison checks the expected excerpts and original page numbers without calling Groq.

```powershell
& .\.venv\Scripts\python.exe .\backend\scripts\evaluate_demo_retrieval.py --engine local
& .\.venv-demo\Scripts\python.exe .\backend\scripts\evaluate_demo_retrieval.py --engine demo
& .\.venv-demo\Scripts\python.exe .\backend\scripts\evaluate_demo_retrieval.py --engine demo-lexical
```

Add `--json` for the ranked pages, chunk IDs, and scores. The local evaluation uses an isolated Chroma index and needs the MiniLM weights in its Hugging Face cache.

| Measure | Local MiniLM + Chroma | Public MiniLM + ONNX | Earlier lexical demo |
| --- | ---: | ---: | ---: |
| Simple questions, complete evidence | 3/3 | 3/3 | 3/3 |
| English paraphrases, complete evidence | 1/3 | 3/3 | 1/3 |
| Cross-page questions, complete evidence | 3/3 | 3/3 | 1/3 |
| Arabic single-page questions, complete evidence | 3/3 | 3/3 | 1/3 |
| Unsupported questions, no context returned | 1/3 | 1/3 | 1/3 |

The public demo found complete evidence for 12 of 12 answerable questions, and the local engine found 10 of 12. The public and local engines ranked the expected page first in 9 of 9 and 7 of 9 single-page questions, respectively. The cross-page cases included one Arabic question.

The original public cosine scores for the pilot team and workshop location passages were 0.2927 and 0.2235, below the 0.30 cutoff, with zero exact-term score. After an empty result, the public backend retries once with simpler wording for two common English size and location question forms. It keeps the original subject, cutoff, and exact-term scoring. The expected passages scored 0.3090 and 0.3292 on the retry. All four additional unsupported size and location controls returned no passages.

### Unsupported questions and Groq verification

The catering-budget question returned no context and skipped Groq. The launch-time and Arabic workshop-fee questions retrieved related passages. The mocked SDK tests verify a citation-free refusal in either language and reject invented figures cited to genuine but unrelated passages.

Two live `openai/gpt-oss-20b` calls used the operator-confirmed Free key and synthetic sample only. Groq answered the trial-session venue question, and the API verified the citation on page 2. Groq returned zero claims and `insufficient_context` for the launch-time question despite receiving pages 1 and 2. The app returned no citations for that refusal. The offline evaluation covers one two-page PDF. Citation verification checks quote provenance and figures, so a nonnumeric claim attached to an unrelated genuine quote can still pass if the model answers incorrectly.

## Resource limits and memory

| Windows process memory | Peak working set |
| --- | ---: |
| Full local backend during MiniLM indexing and chat | 1,203.9 MiB |
| ONNX prototype with the JSON tokenizer | 544.0 MiB |
| Demo with ONNX and native SentencePiece, final validation | 328.2 MiB |

The demo ran in a clean Python environment from `requirements-demo.txt`. In two Windows cold starts, one Uvicorn worker reached health in 2.31 to 2.32 seconds at 243.5 to 244.1 MiB resident. It used up to 250.9 MiB after three concurrent chats, including Arabic, cross-page, and 600-character requests. All three finished in 0.25 to 0.26 seconds with the real encoder and the Groq SDK's HTTP transport mocked. Windows reported 549.8 MiB peak commit in the larger run.

The Windows workstation did not have Docker, Podman, or a usable WSL distribution. Linux startup, resident memory, and concurrent-request performance under a cap remain pending. Before public release, use a local validation container on a Linux-capable workstation or check the first authorized Render Free instance under load.

For the optional document-wide memory regression on Windows or Linux, set `ASKMYDOCS_DEMO_PYTHON` to the clean demo interpreter prepared above. Run `pytest backend/tests/test_demo_document_wide_runtime.py` from the local test environment. The probe loads the real demo assets, runs three concurrent broad questions with offline generation, and checks peak resident memory against 512 MiB.

The Windows Uvicorn follow-up with concurrent English summary, Arabic summary, and date questions peaked at 328.2 MiB working set and 549.6 MiB commit across two runs. Generation was offline. The demo's broad paths add no runtime dependency.

For a local container check, use the Render build and start commands with a Python 3.12 x86-64 Linux image built from only the demo source and prepared model files. Apply `--platform linux/amd64 --memory=512m --memory-swap=512m --cpus=0.1`. Time process launch to `/api/health` and send three concurrent chats with a local mock of Groq. Record `VmRSS` and `VmHWM` from the Uvicorn process's `/proc/<pid>/status`, plus `memory.current`, `memory.peak`, and `memory.events` from the container's cgroup. Confirm `memory.max=536870912` and no OOM kills. [Docker resource limits](https://docs.docker.com/engine/containers/resource_constraints/) · [Linux cgroup v2 memory metrics](https://docs.kernel.org/admin-guide/cgroup-v2.html)

Render Free provides 512 MB of RAM and 0.1 CPU. The public process runs the quantized model through ONNX Runtime on CPU and loads the prepared, read-only corpus from its build. Render can restart it from the build files without a persistent disk. [Render compute plans](https://render.com/docs/compute-plans)

Use Uvicorn's startup banner to identify the Windows process ID. Run `Get-Process -Id <process-id> | Select-Object Id,WorkingSet64,PeakWorkingSet64` while exercising the API. These Windows process measurements are estimates for sizing. Render metrics will show the hosted Linux process after a future authorized deployment.

Render Free idles after 15 minutes and ordinarily needs about a minute for the next request. The frontend shows a startup message and lets visitors retry. Avoid keep-alive jobs. Render's free hours, bandwidth, build allowance, and outbound traffic policy can interrupt a portfolio demo. Without a payment method, quota exhaustion can suspend service or block builds. Vercel Hobby also stops serving affected features when free usage is exhausted. [Render Free limits](https://render.com/docs/free) · [Vercel Hobby limits](https://vercel.com/docs/plans/hobby)

The limiter runs in process memory. A service restart or scale-out resets or splits its counters, including the rolling daily cap. Keep one worker on Render. Per-window caps cover verification attempts, and the daily cap counts admitted chats after verification when enabled. Cloudflare validation helps limit automated traffic. Exact CORS controls browser access, while the backend limiter and optional challenge also protect direct API calls. Groq still enforces its own request and token budgets, which can return 429 before the app's limits. The UI preserves the draft and observes `Retry-After` when supplied.

## Troubleshooting and rollback

- **Render asks for a card.** Stop that setup. Vercel Hobby remains the static frontend host. Running this API in Vercel Functions would require a separate packaging and runtime check for the encoder and native wheels, plus a strategy for rate limits across instances. Vercel documents a standard 500 MB uncompressed limit for Python functions and a 100 MB source-upload limit for Hobby deployments through its CLI. [Vercel Python runtime](https://vercel.com/docs/functions/runtimes/python) · [Vercel limits](https://vercel.com/docs/limits)
- **Demo API cannot start.** Confirm `DEMO_MODE=true`, the `backend` root, and the full demo build command. Run `prepare_demo_model.py --check` with the demo interpreter and verify that `app/demo_data/` includes the generated corpus and embeddings. A dependency error mentioning Chroma usually means the local branch was selected.
- **Browser cannot connect.** Check the service directly at `/api/health`. Wait for an idle service to start, confirm the Vercel build's `VITE_API_URL`, and check the exact browser origin in Render. If the demo is rate-limited, allow the indicated delay.
- **Groq stops responding or reports 429.** Verify the key and plan. Wait for quotas to reset or lower the app limits to spread traffic. Keep the service on Free.
- **Source recovery or rollback.** Revert to the previous approved demo build and verify `demo_mode: true` and the blocked endpoints before sharing the URL again. Render Free supports rollback to its two most recent previous deploys. For local development, restart the full environment with `DEMO_MODE=false` and point Vite at `http://127.0.0.1:8000`. Local uploads, index and chat databases remain managed by the local runtime. [Render rollbacks and free limits](https://render.com/docs/free)
