{
    'name': 'Ask Odoo',
    'version': '1.0',
    'summary': 'A custom module scaffolded by AI',
    'description': 'Basic module with dependencies on stock and account. Includes AI Chat Client Action.',
    'category': 'Custom',
    'author': 'Antigravity',
    'depends': ['base', 'stock', 'account', 'web'],
    'data': [
        'security/ir.model.access.csv',
        'views/odoo_ai_chatbot_views.xml',
    ],
    'assets': {
        'web.assets_backend': [
            'ask_odoo/static/src/components/ai_chat/ai_chat.scss',
            'ask_odoo/static/src/components/ai_chat/ai_chat.xml',
            'ask_odoo/static/src/components/ai_chat/ai_chat.js',
        ],
    },
    'installable': True,
    'application': True,
    'auto_install': False,
}
