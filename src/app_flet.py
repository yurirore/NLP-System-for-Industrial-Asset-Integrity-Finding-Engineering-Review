"""
Flet Desktop App — Equipment Maintenance NLP Pipeline
======================================================

Minimalist black-and-white design following the Minimal Design System philosophy.

Usage:
    flet run src/app_flet.py
"""

import os
import sys
import threading
import time
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flet as ft
import pandas as pd

from src.preprocessing.nltk_setup import setup_nltk
from src.utils.config import NLTK_DIR, SAMPLE_DATA_DIR


# ── Required columns ───────────────────────────────────────────────────

REQUIRED_COLUMNS = {"DESCRIPTION", "PRIORITY", "INITIAL RECOMMENDATION"}


# ── Design Tokens ───────────────────────────────────────────────────────

class DS:
    BG = "#ffffff"
    FG = "#000000"
    MUTED = "#737373"
    BORDER = "#e5e7eb"
    HOVER = "#f5f5f5"
    PADDING = 32
    PADDING_SM = 16
    GAP = 24
    GAP_SM = 12
    MAX_WIDTH = 760


# ── Pipeline State ─────────────────────────────────────────────────────

class PipelineState:
    models = None
    loaded = False
    loading = False

    @classmethod
    def load_async(cls, on_done, on_error):
        if cls.loaded:
            on_done()
            return
        if cls.loading:
            # Already loading — poll until finished
            def _wait():
                while cls.loading:
                    time.sleep(0.1)
                if cls.models is not None:
                    on_done()
                else:
                    on_error("Model loading failed")
            threading.Thread(target=_wait, daemon=True).start()
            return
        cls.loading = True
        def _load():
            try:
                setup_nltk(NLTK_DIR)
                from src.pipeline import load_all_models
                cls.models = load_all_models()
                cls.loaded = True
                cls.loading = False
                on_done()
            except Exception as e:
                cls.loading = False
                on_error(str(e))
        threading.Thread(target=_load, daemon=True).start()

    @classmethod
    def load_sync(cls):
        """Synchronous model loading — blocks until done. Used with asyncio.to_thread."""
        if cls.loaded:
            return
        if cls.loading:
            while cls.loading:
                time.sleep(0.1)
            return
        cls.loading = True
        try:
            setup_nltk(NLTK_DIR)
            from src.pipeline import load_all_models
            cls.models = load_all_models()
            cls.loaded = True
        except Exception as e:
            raise e
        finally:
            cls.loading = False


# ── Helpers ─────────────────────────────────────────────────────────────

def label(text, size=14, color=DS.FG, bold=False, muted=False):
    return ft.Text(text, size=size, color=DS.MUTED if muted else color,
                   weight=ft.FontWeight.BOLD if bold else ft.FontWeight.NORMAL)

def card(content, pad=DS.PADDING):
    return ft.Container(content=content, padding=pad,
                        border=ft.Border.all(1, DS.BORDER), border_radius=8, bgcolor=DS.BG)

def divider():
    return ft.Container(height=1, bgcolor=DS.BORDER)

def spacer(h=DS.GAP_SM):
    return ft.Container(height=h)


# ── CSV Validation ──────────────────────────────────────────────────────

def validate_csv_columns(df: pd.DataFrame) -> tuple:
    """Returns (is_valid: bool, message: str)."""
    actual = set(df.columns)
    if actual != REQUIRED_COLUMNS:
        return False, (
            "Please make sure your file should only include 3 columns: "
            "DESCRIPTION, PRIORITY, and INITIAL RECOMMENDATION only."
        )
    return True, f"✅ Validated — {len(df)} records loaded"


# ── App ─────────────────────────────────────────────────────────────────

def main(page: ft.Page):
    page.title = "Helparooni - Your Engineering Review Helper"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor = DS.BG
    page.padding = 0
    page.window_width = 900
    page.window_height = 720
    page.window_min_width = 640
    page.window_min_height = 480

    main_content = ft.Column([], spacing=0)
    loaded_df = [None]  # mutable closure

    # ── File Picker ─────────────────────────────────────────────────
    file_picker = ft.FilePicker()

    # ── CSV Loading & Validation ────────────────────────────────────
    dataset_status = ft.Text("", size=14, color=DS.MUTED, visible=False)
    dataset_error = ft.Text("", size=14, color=DS.FG, visible=False)
    preview_table = ft.Container(visible=False, padding=0)

    def load_csv(file_path: str):
        dataset_error.visible = False
        dataset_status.visible = False
        preview_table.visible = False

        if not os.path.exists(file_path):
            dataset_error.value = f"❌ File not found:\n{file_path}"
            dataset_error.visible = True
            page.update()
            return

        if not file_path.lower().endswith(".csv"):
            dataset_error.value = "❌ Please select a .csv file."
            dataset_error.visible = True
            page.update()
            return

        try:
            df = pd.read_csv(file_path)
            is_valid, msg = validate_csv_columns(df)
            if not is_valid:
                dataset_error.value = "❌ " + msg
                dataset_error.visible = True
                page.update()
                return

            loaded_df[0] = df
            dataset_status.value = f"✅ {os.path.basename(file_path)} — {msg}"
            dataset_status.visible = True

            preview_rows = []
            for i, (_, row) in enumerate(df.head(5).iterrows()):
                desc = str(row.get("DESCRIPTION", ""))
                prio = str(row.get("PRIORITY", ""))
                preview_rows.append(ft.DataRow([
                    ft.DataCell(ft.Text(desc, size=12, color=DS.FG)),
                    ft.DataCell(ft.Text(prio, size=12, color=DS.FG)),
                ]))

            preview_table.content = ft.Column([
                spacer(4),
                label(f"Preview (first 5 of {len(df)} rows):", size=13, bold=True),
                spacer(4),
                ft.Row([
                    ft.DataTable(
                        columns=[
                            ft.DataColumn(ft.Text("Description", size=11, color=DS.MUTED)),
                            ft.DataColumn(ft.Text("Priority", size=11, color=DS.MUTED)),
                        ],
                        rows=preview_rows,
                        border=ft.Border.all(1, DS.BORDER),
                        border_radius=6,
                        heading_row_color=DS.HOVER,
                        expand=True,
                    ),
                ], scroll=ft.ScrollMode.AUTO, expand=True),
            ], spacing=0)
            preview_table.visible = True

        except Exception as ex:
            dataset_error.value = f"❌ Error reading file:\n{ex}"
            dataset_error.visible = True

        page.update()

    # ── Build shell ─────────────────────────────────────────────────
    def build_shell():
        header = ft.Container(
            content=ft.Row([
                ft.Text("🛠️", size=20),
                ft.Container(width=8),
                ft.Text("HELPAROONI", size=18, weight=ft.FontWeight.BOLD, color=DS.FG),
            ]),
            padding=ft.Padding(left=DS.PADDING, right=DS.PADDING, top=DS.PADDING_SM, bottom=DS.PADDING_SM),
        )

        # ── Processing widgets ──────────────────────────────────────
        processing_spinner = ft.ProgressRing(width=20, height=20, stroke_width=2, visible=False)
        processing_status = ft.Text("", size=13, color=DS.MUTED, visible=False)
        processing_error = ft.Text("", size=13, color=DS.FG, visible=False)
        results_table = ft.Container(visible=False, padding=0)
        export_btn = ft.Container(visible=False, padding=0)  # shown after results
        stored_results = [None]  # keep results data for export
        btn_icon = ft.Text("▶️", size=16)
        btn_label = ft.Text("Start Batch Processing", size=14, weight=ft.FontWeight.BOLD, color=DS.BG)
        run_btn = ft.Container(
            content=ft.Row([btn_icon, ft.Container(width=6), btn_label]),
            padding=ft.Padding(left=DS.PADDING, right=DS.PADDING, top=14, bottom=14),
            bgcolor=DS.FG, border_radius=6,
        )

        # ── Batch Processing ────────────────────────────────────────
        async def run_batch(e):
            df = loaded_df[0]
            if df is None:
                processing_error.value = "Please load a dataset first."
                processing_error.visible = True
                results_table.visible = False
                page.update()
                return

            processing_error.visible = False
            processing_spinner.visible = True
            processing_status.visible = True
            processing_status.value = "⏳ Initializing models..."
            btn_icon.value = "⏳"
            btn_label.value = "Processing..."
            btn_label.color = DS.MUTED
            page.update()

            try:
                loop = asyncio.get_event_loop()
                # Load models if needed
                if not PipelineState.loaded:
                    processing_status.value = "⏳ Loading models..."
                    page.update()
                    await loop.run_in_executor(None, PipelineState.load_sync)

                if not PipelineState.loaded:
                    return

                # Run batch processing — one item at a time with progress
                m = PipelineState.models
                total = len(df)
                from src.pipeline import run_complete_pipeline

                results = []
                for idx, row in df.iterrows():
                    processing_status.value = f"⏳ Processing {idx + 1}/{total}..."
                    page.update()

                    result = await loop.run_in_executor(
                        None,
                        lambda r=row: run_complete_pipeline(
                            description=r['DESCRIPTION'],
                            priority=str(r['PRIORITY']),
                            initial_recommendation=r['INITIAL RECOMMENDATION'],
                            classification_model=m[0], label_encoder=m[1],
                            summarization_model=m[2], summarization_tokenizer=m[3],
                            embedder=m[4], centroids=m[5],
                            verbose=False,
                        ),
                    )
                    results.append(result)

                # All UI code below runs on the main async thread
                processing_spinner.visible = False
                processing_status.value = f"✅ Processed {len(results)} items."
                btn_icon.value = "▶️"
                btn_label.value = "Start Batch Processing"
                btn_label.color = DS.BG

                stored_results[0] = results
                data_rows = []
                for i, r in enumerate(results):
                    data_rows.append(ft.DataRow([
                        ft.DataCell(ft.Text(str(r['original_input']['description']), size=12, color=DS.FG)),
                        ft.DataCell(ft.Text(str(r['original_input']['initial_recommendation']), size=12, color=DS.MUTED)),
                        ft.DataCell(ft.Text(r['predicted_recommendation'], size=12, color=DS.FG)),
                        ft.DataCell(ft.Text(r['structured_output'], size=12, color=DS.FG, selectable=True)),
                    ]))

                # Build export button
                async def _export(_):
                    path = await file_picker.save_file(
                        dialog_title="Save Results",
                        file_name="pipeline_results.csv",
                        allowed_extensions=["csv"],
                    )
                    if path and not path.endswith(".csv"):
                        path += ".csv"
                    if path:
                        import pandas as pd
                        from src.utils.io_utils import save_results_to_csv
                        save_results_to_csv(stored_results[0], path)

                export_btn.content = ft.Container(
                    content=ft.Row([
                        ft.Text("💾", size=16),
                        ft.Container(width=6),
                        ft.Text("Export Results", size=14, weight=ft.FontWeight.BOLD, color=DS.BG),
                    ]),
                    padding=ft.Padding(left=DS.PADDING, right=DS.PADDING, top=12, bottom=12),
                    bgcolor=DS.FG, border_radius=6,
                    on_click=_export,
                )
                export_btn.visible = True

                results_table.content = ft.Column([
                    spacer(DS.GAP_SM),
                    label("📋 Results (" + str(len(results)) + " items)", size=18, bold=True),
                    spacer(8),
                    ft.Row([
                        ft.DataTable(
                            columns=[
                                ft.DataColumn(ft.Text("DESCRIPTION", size=11, color=DS.MUTED)),
                                ft.DataColumn(ft.Text("INITIAL RECOMMENDATION", size=11, color=DS.MUTED)),
                                ft.DataColumn(ft.Text("PREDICTED RECOMMENDATION", size=11, color=DS.MUTED)),
                                ft.DataColumn(ft.Text("MAINTENANCE ACTION PLAN", size=11, color=DS.MUTED)),
                            ],
                            rows=data_rows,
                            border=ft.Border.all(1, DS.BORDER),
                            border_radius=6,
                            heading_row_color=DS.HOVER,
                            expand=True,
                        ),
                    ], scroll=ft.ScrollMode.AUTO, expand=True),
                    spacer(DS.GAP_SM),
                    export_btn,
                ], spacing=0)
                results_table.visible = True

            except Exception as ex:
                processing_spinner.visible = False
                processing_status.value = f"❌ Error: {ex}"
                btn_icon.value = "▶️"
                btn_label.value = "Start Batch Processing"
                btn_label.color = DS.BG
            page.update()

        run_btn.on_click = run_batch

        # ── Main content ────────────────────────────────────────────
        load_btn = ft.Container(
            content=ft.Row([
                ft.Text("📂", size=16),
                ft.Container(width=6),
                ft.Text("Load Dataset", size=14, weight=ft.FontWeight.BOLD, color=DS.BG),
            ]),
            padding=ft.Padding(left=DS.PADDING, right=DS.PADDING, top=12, bottom=12),
            bgcolor=DS.FG, border_radius=6,
        )
        async def _pick_file(_):
            files = await file_picker.pick_files(
                allow_multiple=False,
                allowed_extensions=["csv"],
                dialog_title="Select CSV File",
            )
            if files and len(files) > 0:
                load_csv(files[0].path)
        load_btn.on_click = _pick_file

        main_content.controls = [
            ft.Container(
                content=ft.Column([
                    spacer(24),
                    ft.Text("Welcome to Helparooni,", size=28, weight=ft.FontWeight.BOLD, color=DS.FG),
                    ft.Text("your Engineering Review Helper!", size=28, weight=ft.FontWeight.BOLD, color=DS.FG),
                    spacer(8),
                    ft.Text(
                        spans=[
                            ft.TextSpan("I am an AI-assisted tool to help re-classify "),
                            ft.TextSpan("Maintenance Risks", style=ft.TextStyle(italic=True)),
                            ft.TextSpan(" for the findings. Additionally, I will also generate "),
                            ft.TextSpan("Maintenance Action Plan", style=ft.TextStyle(italic=True)),
                            ft.TextSpan(" for "),
                            ft.TextSpan("Engineering Review", style=ft.TextStyle(italic=True)),
                            ft.TextSpan(" reference purpose!"),
                        ],
                        size=15, color=DS.MUTED,
                    ),
                    spacer(DS.GAP),

                    label("📁 Asset Integrity Findings", size=18, bold=True),
                    spacer(DS.GAP_SM),
                    card(ft.Column([
                        load_btn,
                        spacer(8),
                        ft.Text(".csv format only", size=12, color=DS.MUTED, italic=True),
                    ], spacing=0)),
                    spacer(DS.GAP_SM),
                    dataset_status,
                    dataset_error,
                    preview_table,

                    spacer(DS.GAP),
                    run_btn,
                    spacer(8),
                    ft.Row([processing_spinner, processing_status], spacing=8,
                           vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    processing_error,
                    results_table,
                    spacer(40),
                ], spacing=0),
                padding=ft.Padding(left=DS.PADDING, right=DS.PADDING, top=0, bottom=0),
            )
        ]

        # ── Assembly with scroll ─────────────────────────────────────
        body_stack = ft.Column([main_content], spacing=0)

        page.controls = [
            ft.Column([
                header,
                divider(),
                ft.Container(
                    content=body_stack,
                    expand=True,
                    padding=0,
                ),
            ], spacing=0, expand=True, scroll=ft.ScrollMode.AUTO),
        ]
        page.update()

    build_shell()


ft.app(target=main)  # nosec: deprecated but works; no run() alternative available
