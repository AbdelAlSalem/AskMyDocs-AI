import { useEffect, useRef, useState } from 'react'
import { CircleAlert, LoaderCircle, ShieldCheck } from 'lucide-react'

interface Turnstile {
  render(container: HTMLElement, options: {
    sitekey: string
    action: string
    theme: 'dark'
    size: 'flexible' | 'compact'
    appearance: 'interaction-only'
    callback: (token: string) => void
    'error-callback': () => true
    'expired-callback': () => void
    'timeout-callback': () => void
  }): string
  reset(widgetId: string): void
  remove(widgetId: string): void
}

declare global {
  interface Window {
    turnstile?: Turnstile
    askMyDocsTurnstileReady?: () => void
  }
}

let loadingScript: Promise<Turnstile> | undefined

function loadTurnstile(): Promise<Turnstile> {
  if (window.turnstile) return Promise.resolve(window.turnstile)
  if (!loadingScript) {
    loadingScript = new Promise<Turnstile>((resolve, reject) => {
      const script = document.createElement('script')
      const timeout = window.setTimeout(fail, 20_000)

      function fail() {
        window.clearTimeout(timeout)
        delete window.askMyDocsTurnstileReady
        script.remove()
        reject(new Error('Browser verification could not load.'))
      }

      window.askMyDocsTurnstileReady = () => {
        if (!window.turnstile) {
          fail()
          return
        }
        window.clearTimeout(timeout)
        delete window.askMyDocsTurnstileReady
        resolve(window.turnstile)
      }
      script.src = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit&onload=askMyDocsTurnstileReady'
      script.async = true
      script.onerror = fail
      document.head.append(script)
    }).catch((failure: unknown) => {
      loadingScript = undefined
      throw failure
    })
  }
  return loadingScript
}

export function BrowserVerification({ siteKey, action, cycle, onToken }: {
  siteKey: string
  action: string
  cycle: number
  onToken: (token?: string) => void
}) {
  const container = useRef<HTMLDivElement>(null)
  const [status, setStatus] = useState<'checking' | 'verified' | 'unavailable'>('checking')
  const [retry, setRetry] = useState(0)

  useEffect(() => {
    let mounted = true
    let turnstile: Turnstile | undefined
    let widgetId: string | undefined
    const element = container.current
    setStatus('checking')
    onToken(undefined)

    function failed() {
      if (mounted) {
        setStatus('unavailable')
        onToken(undefined)
      }
    }

    void loadTurnstile().then((loaded) => {
      if (!mounted || !element) return
      turnstile = loaded
      widgetId = loaded.render(element, {
        sitekey: siteKey,
        action,
        theme: 'dark',
        size: element.clientWidth < 300 ? 'compact' : 'flexible',
        appearance: 'interaction-only',
        callback: (token) => {
          if (!mounted) return
          onToken(token)
          setStatus('verified')
        },
        'error-callback': () => { failed(); return true },
        'expired-callback': () => {
          if (!mounted || !widgetId) return
          onToken(undefined)
          setStatus('checking')
          loaded.reset(widgetId)
        },
        'timeout-callback': failed,
      })
    }).catch(failed)

    return () => {
      mounted = false
      if (widgetId) turnstile?.remove(widgetId)
    }
  }, [action, cycle, onToken, retry, siteKey])

  return (
    <div className="browser-verification">
      <div ref={container} />
      <div className="verification-status" role="status">
        {status === 'checking' && <><LoaderCircle size={14} className="spin" aria-hidden="true" /> Verifying browser…</>}
        {status === 'verified' && <><ShieldCheck size={14} aria-hidden="true" /> Browser verified</>}
        {status === 'unavailable' && (
          <><CircleAlert size={14} aria-hidden="true" /> Verification unavailable
            <button type="button" className="text-button" onClick={() => setRetry((current) => current + 1)}>Try again</button>
          </>
        )}
      </div>
    </div>
  )
}
