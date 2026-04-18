from odoo import models
import logging
import pandas as pd

_logger = logging.getLogger(__name__)

class AskOdooModel(models.Model):
    _inherit = 'ask.odoo.model'

    def _normalize_column(self, df, key):
        """Find the closest matching column name, case-insensitive."""
        key_lower = str(key).lower().replace(' ', '_')
        for col in df.columns:
            if str(col).lower().replace(' ', '_') == key_lower:
                return col
        return None

    def _generate_explicit_chart(self, df, chart_config):
        """
        Generates chart data based on explicit chart_config requested by the user.
        Raises ValueError or returns None if the structure is invalid.
        """
        try:
            if df is None or df.empty or not isinstance(chart_config, dict):
                return None

            chart_type = chart_config.get('type', 'bar')
            x_col_key = chart_config.get('x')
            y_col_keys = chart_config.get('y', [])
            
            if not isinstance(y_col_keys, list):
                y_col_keys = [y_col_keys]

            if not x_col_key or not y_col_keys:
                _logger.warning("AskOdoo: chart_config missing 'x' or 'y' keys.")
                return None

            x_col = self._normalize_column(df, x_col_key)
            if not x_col:
                _logger.warning(f"AskOdoo: X column '{x_col_key}' not found in DataFrame.")
                return None
                
            labels = df[x_col].astype(str).tolist()

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
            for i, y_col_key in enumerate(y_col_keys):
                y_col = self._normalize_column(df, y_col_key)
                if not y_col:
                    continue
                    
                # Convert values, coercing errors to 0
                values = pd.to_numeric(df[y_col], errors='coerce').fillna(0).tolist()

                dataset = {
                    'label': str(y_col),
                    'data': values,
                }

                if chart_type in ['pie', 'doughnut']:
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

            if not datasets:
                _logger.warning("AskOdoo: No valid Y columns found for chart.")
                return None

            chart_data = {
                'type': chart_type,
                'labels': labels,
                'datasets': datasets,
                'title': f"{', '.join(str(ds['label']) for ds in datasets)} by {x_col}",
            }

            _logger.info(f"AskOdoo: Explicit chart generated — type={chart_type}, labels={len(labels)}, datasets={len(datasets)}")
            return chart_data

        except Exception as e:
            _logger.warning(f"AskOdoo: Explicit chart generation failed: {e}")
            return None

