"""
chat_history.py - SQLite chat history module for Elite Multi-Doc RAG.
"""
from __future__ import annotations

import html
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = "chat_history.db"


class ChatHistory:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                source      TEXT DEFAULT '',
                source_file TEXT DEFAULT '',
                timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id)")
        conn.commit()
        conn.close()

    @staticmethod
    def _clean(text: str) -> str:
        lines = [
            re.sub(r"[ \t]+", " ", line).strip()
            for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        ]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)

    def save_exchange(self, session_id: str, question: str, answer: str, source: str = "", source_file: str = ""):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO messages (session_id, role, content, source, source_file) VALUES (?,?,?,?,?)",
            (session_id, "user", self._clean(question), "", "")
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, source, source_file) VALUES (?,?,?,?,?)",
            (session_id, "bot", self._clean(answer), source, source_file)
        )
        conn.commit()
        conn.close()

    def get_session_history(self, session_id: str) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT role, content, source, source_file, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
        """, (session_id,))
        rows = cur.fetchall()
        conn.close()
        return [
            {"role": r[0], "content": r[1], "source": r[2], "source_file": r[3], "timestamp": r[4]}
            for r in rows
        ]

    def get_all_sessions(self) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT session_id, COUNT(*) as msg_count,
                   MIN(timestamp) as started, MAX(timestamp) as last_msg
            FROM messages
            GROUP BY session_id
            ORDER BY last_msg DESC
        """)
        rows = cur.fetchall()
        conn.close()
        return [
            {"session_id": r[0], "messages": r[1], "started": r[2], "last_msg": r[3]}
            for r in rows
        ]

    def clear_session(self, session_id: str) -> int:
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        deleted = conn.total_changes
        conn.commit()
        conn.close()
        return deleted

    def clear_all(self) -> int:
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM messages")
        deleted = conn.total_changes
        conn.commit()
        conn.close()
        return deleted

    def export_as_txt(self, session_id: str) -> str:
        messages = self.get_session_history(session_id)
        if not messages:
            return "No messages found for this session."
        lines = [
            "=" * 72,
            "  ELITE MULTI-DOC RAG - CHAT EXPORT",
            "=" * 72,
            f"Session ID : {session_id}",
            f"Exported   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Messages   : {len(messages)}",
            "=" * 72,
            "",
        ]
        for msg in messages:
            time_str = msg["timestamp"][11:19] if msg["timestamp"] else ""
            if msg["role"] == "user":
                lines += [f"[{time_str}] YOU:", f"  {msg['content']}", ""]
            else:
                tag = f" [{msg['source'].upper()}]" if msg["source"] else ""
                lines += [f"[{time_str}] ASSISTANT{tag}:", f"  {msg['content']}", ""]
                if msg["source_file"]:
                    lines += [f"  Sources: {msg['source_file']}", ""]
        return "\n".join(lines)

    def export_as_json(self, session_id: str) -> str:
        messages = self.get_session_history(session_id)
        return json.dumps({
            "session_id": session_id,
            "exported_at": datetime.now().isoformat(),
            "total_messages": len(messages),
            "messages": messages,
        }, indent=2, ensure_ascii=False)

    def export_as_html(self, session_id: str) -> str:
        messages = self.get_session_history(session_id)
        html_doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Elite Multi-Doc RAG Chat Export</title>
<style>
body{{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;padding:20px;background:#f5f7fa;color:#111827}}
.header{{background:#111827;color:white;padding:22px;border-radius:12px;margin-bottom:24px}}
.message{{margin-bottom:16px;display:flex;flex-direction:column}}.message.user{{align-items:flex-end}}.message.bot{{align-items:flex-start}}
.bubble{{max-width:75%;padding:12px 16px;border-radius:16px;font-size:14px;line-height:1.5;white-space:pre-wrap}}
.user .bubble{{background:#1a56db;color:white;border-bottom-right-radius:4px}}.bot .bubble{{background:white;border:1px solid #e2e8f0;border-bottom-left-radius:4px}}
.meta{{font-size:11px;color:#94a3b8;margin-top:4px}}.source{{font-size:12px;color:#64748b;margin-top:8px}}
</style></head><body>
<div class="header"><h1>Elite Multi-Doc RAG - Chat Export</h1><p>Session: {html.escape(session_id)} | Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Messages: {len(messages)}</p></div>
"""
        for msg in messages:
            role = "user" if msg["role"] == "user" else "bot"
            content = html.escape(msg["content"])
            time_str = msg["timestamp"][11:19] if msg["timestamp"] else ""
            source = ""
            if msg["role"] != "user" and msg["source_file"]:
                source = f"<div class='source'><b>Sources:</b> {html.escape(msg['source_file'])}</div>"
            html_doc += f"<div class='message {role}'><div class='bubble'>{content}{source}</div><div class='meta'>{time_str}</div></div>"
        html_doc += "</body></html>"
        return html_doc
