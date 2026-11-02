from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bank_distress_bot import BASE_DIR, CANONICAL, DEFAULT_CSV, DEFAULT_MODEL, Backend

INTENT_MODEL = BASE_DIR / "models" / "intent_model.joblib"
NOT_UNDERSTOOD = "I didn't quite catch that"

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Financial-Distress Early-Warning Assistant</title><style>
:root{--bg:#f4f6f8;--card:#fff;--ink:#1d2733;--mut:#5b6b7a;--acc:#0b5cab;--bot:#e8eef5;--me:#0b5cab}
@media(prefers-color-scheme:dark){:root{--bg:#12171d;--card:#1b232c;--ink:#e6edf3;--mut:#9aa9b8;--acc:#4da3ff;--bot:#26313d;--me:#1f5fa8}}
*{box-sizing:border-box}body{margin:0;font-family:Arial,Helvetica,sans-serif;background:var(--bg);color:var(--ink);display:flex;justify-content:center}
main{width:100%;max-width:860px;height:100vh;display:flex;flex-direction:column;padding:12px}
header{display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:space-between}
h1{font-size:1.15rem;margin:4px 0}#sum{font-size:.8rem;color:var(--mut)}
#log{flex:1;overflow-y:auto;background:var(--card);border-radius:10px;padding:12px;margin:8px 0;display:flex;flex-direction:column;gap:8px}
.m{padding:8px 12px;border-radius:12px;max-width:88%;white-space:pre-wrap;line-height:1.4;font-size:.92rem}
.bot{background:var(--bot);align-self:flex-start}.me{background:var(--me);color:#fff;align-self:flex-end}
#chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px}
button,input{font:inherit}button.chip{border:1px solid var(--acc);background:transparent;color:var(--acc);border-radius:16px;padding:4px 10px;cursor:pointer;font-size:.8rem}
form{display:flex;gap:6px}#q{flex:1;padding:10px;border-radius:8px;border:1px solid var(--mut);background:var(--card);color:var(--ink)}
.btn{border:0;border-radius:8px;padding:0 14px;background:var(--acc);color:#fff;cursor:pointer}.btn.sm{padding:5px 12px;font-size:.8rem;margin-left:8px}

footer{font-size:.72rem;color:var(--mut);margin-top:6px}
</style></head><body><main>
<header><h1>Financial-Distress Early-Warning Assistant</h1>
<button type="button" class="btn sm" id="clear" title="Clear the conversation">Clear</button></header>
<div id="sum"></div><div id="log" aria-live="polite"></div><div id="chips"></div>
<form id="f"><input id="q" autocomplete="off" placeholder="Ask e.g. Find customers at risk"><button class="btn">Send</button></form>
<footer>Flags are prompts for employee review, not decisions about a customer. All questions are written to the audit log.</footer>
</main><script>
const $=id=>document.getElementById(id), log=$('log');
function add(text,cls){const d=document.createElement('div');d.className='m '+cls;d.textContent=text;log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
async function ask(text){text=text.trim();if(!text)return;add(text,'me');$('q').value='';
 try{const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
 const j=await r.json();add(j.reply||j.error,'bot');}catch(e){add('Could not reach the server.','bot')}}
$('f').onsubmit=e=>{e.preventDefault();ask($('q').value)};
['Find customers at risk','Find customers already in financial distress','Find customers with late payments','Show customers with multiple warning signs','How does flagging work?','help']
 .forEach(t=>{const b=document.createElement('button');b.type='button';b.className='chip';b.textContent=t;b.onclick=()=>ask(t);$('chips').appendChild(b)});
function welcome(){fetch('/api/summary').then(r=>r.json()).then(j=>{$('sum').textContent=j.summary;add(j.banner+'\\nSay "help" for examples.','bot')})}
welcome();
$('clear').onclick=async()=>{log.textContent='';$('q').value='';try{await fetch('/api/clear',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})}catch(e){}welcome();$('q').focus()};
</script></body></html>"""


class App:
    def __init__(self, be: Backend):
        self.be, self.lock, self.intent = be, threading.Lock(), None
        if INTENT_MODEL.exists():
            try:
                import joblib
                self.intent = joblib.load(INTENT_MODEL)
            except Exception as e:
                print(f"[intent model not loaded: {type(e).__name__}]")

    def rescue(self, text: str) -> str | None:
        if self.intent is None:
            return None
        from nlp_intent import classify
        idx, _, cid = classify(self.intent, text)
        if idx is None:
            return None
        template = CANONICAL[idx][0]
        if "{id}" in template:
            cid = cid or self.be.session.last_customer
            if cid is None:
                return None
            return template.format(id=cid)
        return template

    def clear(self) -> None:
        with self.lock:
            self.be.session = self.be.bot.new_session(self.be.session.employee_id)

    def ask(self, text: str) -> str:
        with self.lock:
            reply = self.be.ask(text)
            if reply.startswith(NOT_UNDERSTOOD):
                cmd = self.rescue(text)
                if cmd:
                    reply = f"(Understood as: \"{cmd}\")\n\n" + self.be.ask(cmd)
            return reply


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _local(self) -> bool:
            host = (self.headers.get("Host") or "").split(":")[0]
            return host in ("127.0.0.1", "localhost")

        def do_GET(self):
            if not self._local():
                return self._json(403, {"error": "forbidden"})
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/api/summary":
                with app.lock:
                    return self._json(200, {"summary": app.be.summary(), "banner": app.be.banner()})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._local() or self.path not in ("/api/ask", "/api/clear"):
                return self._json(403 if not self._local() else 404, {"error": "not allowed"})
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                return self._json(415, {"error": "JSON only"})
            if self.path == "/api/clear":
                self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 4000))
                app.clear()
                return self._json(200, {"ok": True})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > 4000:
                    return self._json(413, {"error": "message too long"})
                d = json.loads(self.rfile.read(n) or b"{}")
                text = str(d.get("text", ""))[:500].strip()
            except (ValueError, TypeError):
                return self._json(400, {"error": "bad request"})
            if not text:
                return self._json(400, {"error": "empty message"})
            self._json(200, {"reply": app.ask(text)})

        def log_message(self, *args):
            pass
    return H


def main() -> None:
    ap = argparse.ArgumentParser(description="Browser chat for the early-warning assistant (localhost only)")
    ap.add_argument("csv", nargs="?", default=DEFAULT_CSV)
    ap.add_argument("--db", default="distress_bot.db")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true", help="open the browser automatically")
    a = ap.parse_args()
    try:
        be = Backend(a.csv, a.db, a.model, None)
    except FileNotFoundError as e:
        raise SystemExit(str(e))
    app = App(be)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(app))
    url = f"http://127.0.0.1:{a.port}"
    print(be.banner())
    print(f"Intent classifier: {'ON' if app.intent else 'off (run nlp_intent.py to enable)'}")
    print(f"Serving on {url}  (localhost only, Ctrl+C to stop)")
    if a.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
