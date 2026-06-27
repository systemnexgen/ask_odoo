{
    'name': 'Ask Odoo',
    'version': '1.0',
    'summary': 'AI module for Odoo ERP',
    'description': 'AI module with dependencies on stock and account. Includes AI Chat Client Action.',
    'category': 'Custom',
    'author': 'System Nexgen',
    'depends': ['base', 'stock', 'account', 'web'],
    'data': [
        'security/ir.model.access.csv',
        'views/odoo_ai_chatbot_views.xml',
    ],
    'assets': {
        'web.assets_backend': [
            'ask_odoo/static/src/lib/chart.umd.min.js',
            'ask_odoo/static/src/components/ai_chat/ai_chat.scss',
            'ask_odoo/static/src/components/ai_chat/ai_chat.xml',
            'ask_odoo/static/src/components/ai_chat/ai_chat.js',
        ],
    },
    'installable': True,
    'application': True,
    'auto_install': False,
}
