from odoo import models
import logging
import pandas as pd

_logger = logging.getLogger(__name__)

class AskOdooModel(models.Model):
    _inherit = 'ask.odoo.model'

    def _detect_chart_data(self, df):
        """
        Analyzes a Pandas DataFrame and returns Chart.js-compatible JSON
        if the data is suitable for visualization. Returns None otherwise.

        Auto-selects chart type:
          - Line chart: if the label column looks like dates/time
          - Pie/Doughnut: if ≤7 categories and only 1 numeric column
          - Bar chart: default for everything else
        """
        try:
            if df is None or df.empty or len(df) < 2:
                return None

            # Classify columns into labels (text/object) and values (numeric)
            label_cols = []
            value_cols = []

            for col in df.columns:
                col_str = str(col)
                # Skip columns named 'id' or ending in '_id' — not useful for charts
                if col_str.lower() == 'id' or col_str.lower().endswith('_id'):
                    continue

                if pd.api.types.is_numeric_dtype(df[col]):
                    value_cols.append(col)
                elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                    label_cols.append(col)

            # Need at least 1 label column and 1 numeric column
            if not label_cols or not value_cols:
                return None

            # Use the first label column as the X-axis
            label_col = label_cols[0]
            labels = df[label_col].astype(str).tolist()

            # Cap at 50 data points for readability
            if len(labels) > 50:
                return None

            # ── Chart Type Selection ──────────────────────────────────
            chart_type = 'bar'  # default

            # Check if labels look like dates/time → line chart
            is_time_series = False
            date_keywords = ['date', 'month', 'year', 'week', 'day', 'time', 'period', 'quarter']
            if any(kw in str(label_col).lower() for kw in date_keywords):
                is_time_series = True
                chart_type = 'line'

            # Small categories with single metric → pie chart
            if len(labels) <= 7 and len(value_cols) == 1 and not is_time_series:
                chart_type = 'doughnut'

            # ── Color Palette ─────────────────────────────────────────
            colors = [
                'rgba(99, 102, 241, 0.8)',    # Indigo
                'rgba(16, 185, 129, 0.8)',    # Emerald
                'rgba(245, 158, 11, 0.8)',    # Amber
                'rgba(239, 68, 68, 0.8)',     # Red
                'rgba(139, 92, 246, 0.8)',    # Violet
                'rgba(6, 182, 212, 0.8)',     # Cyan
                'rgba(236, 72, 153, 0.8)',    # Pink
                'rgba(34, 197, 94, 0.8)',     # Green
                'rgba(251, 146, 60, 0.8)',    # Orange
                'rgba(59, 130, 246, 0.8)',    # Blue
            ]
            border_colors = [c.replace('0.8)', '1)') for c in colors]

            # ── Build Datasets ────────────────────────────────────────
            datasets = []
            for i, vcol in enumerate(value_cols):
                # Convert values, coercing errors to 0
                values = pd.to_numeric(df[vcol], errors='coerce').fillna(0).tolist()

                dataset = {
                    'label': str(vcol),
                    'data': values,
                }

                if chart_type == 'doughnut':
                    # Pie/doughnut: each slice gets a different color
                    dataset['backgroundColor'] = colors[:len(values)]
                    dataset['borderColor'] = border_colors[:len(values)]
                    dataset['borderWidth'] = 2
                else:
                    # Bar/Line: each dataset gets one color
                    color_idx = i % len(colors)
                    dataset['backgroundColor'] = colors[color_idx]
                    dataset['borderColor'] = border_colors[color_idx]
                    dataset['borderWidth'] = 2

                    if chart_type == 'line':
                        dataset['fill'] = False
                        dataset['tension'] = 0.3

                datasets.append(dataset)

            chart_data = {
                'type': chart_type,
                'labels': labels,
                'datasets': datasets,
                'title': f"{', '.join(str(v) for v in value_cols)} by {label_col}",
            }

            _logger.info(f"AskOdoo: Chart detected — type={chart_type}, labels={len(labels)}, datasets={len(datasets)}")
            return chart_data

        except Exception as e:
            _logger.warning(f"AskOdoo: Chart detection failed (non-critical): {e}")
            return None

