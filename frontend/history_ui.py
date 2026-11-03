"""Conversation History tab."""

import gradio as gr
import logging

logger = logging.getLogger(__name__)


def load_history():
    try:
        from backend.main import get_database
        db = get_database()
        if db is None:
            return [], "⚠ Database unavailable."
        records = db.load_history()
        if not records:
            return [], "No saved conversations yet."
        rows = []
        import datetime
        for r in records:
            dt = datetime.datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M")
            rows.append([
                r["id"][:8] + "...",
                dt,
                " ".join(r.get("gloss_sequence", [])),
                r.get("english_text", ""),
                r.get("translated_text", ""),
                r.get("language_name", ""),
            ])
        return rows, f"✅ Loaded {len(rows)} records."
    except Exception as exc:
        logger.error("History load error: %s", exc)
        return [], f"⚠ Error: {exc}"


def delete_all_history():
    try:
        from backend.main import get_database
        db = get_database()
        if db is None:
            return "⚠ Database unavailable."
        count = db.delete_all()
        return f"✅ Deleted {count} records."
    except Exception as exc:
        return f"⚠ Error: {exc}"


def export_transcript():
    try:
        from backend.main import get_database
        db = get_database()
        if db is None:
            return None
        transcript = db.export_transcript()
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(transcript)
            return f.name
    except Exception as exc:
        logger.error("Export error: %s", exc)
        return None


def build_history_tab():
    with gr.Tab("📜 Conversation History"):
        gr.Markdown("""
## Conversation History
Records are saved **only when you click Save** in the Live Translator tab.
No webcam video or microphone audio is stored.
        """)
        with gr.Row():
            btn_load = gr.Button("🔄 Load History", variant="primary")
            btn_delete_all = gr.Button("🗑️ Delete All", variant="stop")
            btn_export = gr.Button("📥 Export Transcript")

        status_box = gr.Textbox(label="Status", interactive=False)

        history_table = gr.Dataframe(
            headers=["ID", "Time", "ISL Glosses", "English", "Translated", "Language"],
            label="Saved Conversations",
            interactive=False,
            wrap=True,
        )

        export_file = gr.File(label="Downloaded Transcript")

        btn_load.click(fn=load_history, outputs=[history_table, status_box])
        btn_delete_all.click(fn=delete_all_history, outputs=[status_box])
        btn_export.click(fn=export_transcript, outputs=[export_file])

