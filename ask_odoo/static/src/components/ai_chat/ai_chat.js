/** @odoo-module **/

import { Component, useState, useRef, useEffect, onWillStart, markup } from "@odoo/owl";
import { useService } from "@web/core/utils/hooks";
import { registry } from "@web/core/registry";
import { Layout } from "@web/search/layout";

export class AiChat extends Component {
    static template = "odoo_ai_chatbot.AiChat";
    static components = { Layout };

    setup() {
        this.orm = useService("orm");
        this.state = useState({
            conversations: [],
            messages: [],
            input: "",
            isTyping: false,
            currentChatId: null,
            darkMode: false,
            sidebarOpen: true,
            currentView: 'chat',
            documents: [], // For Knowledge Base
            chatMode: 'conversation',
            isModeDropdownOpen: false,
            isRefreshingSchema: false,
        });
        this.messagesEndRef = useRef("messagesEnd");
        this.fileInputRef = useRef("fileInput");

        onWillStart(async () => {
            await this.loadConversations();
            await this.loadDocuments();
        });

        useEffect(() => {
            this.scrollToBottom();
        }, () => [this.state.messages.length]);
    }

    scrollToBottom() {
        if (this.messagesEndRef.el) {
            this.messagesEndRef.el.scrollIntoView({ behavior: "smooth" });
        }
    }

    toggleTheme() {
        this.state.darkMode = !this.state.darkMode;
    }

    toggleSidebar() {
        this.state.sidebarOpen = !this.state.sidebarOpen;
    }

    toggleModeDropdown() {
        this.state.isModeDropdownOpen = !this.state.isModeDropdownOpen;
    }

    selectChatMode(mode) {
        this.state.chatMode = mode;
        this.state.isModeDropdownOpen = false;
    }

    async refreshSchema() {
        if (this.state.isRefreshingSchema) return;
        this.state.isRefreshingSchema = true;
        try {
            await this.orm.call("ask.odoo.model", "refresh_schema_index");
            alert("✅ Schema Refreshed Successfully!");
        } catch (e) {
            console.error(e);
            alert("❌ Failed to refresh schema: " + e.message);
        } finally {
            this.state.isRefreshingSchema = false;
            this.state.isModeDropdownOpen = false;
        }
    }

    switchToChat() {
        this.state.currentView = 'chat';
    }

    switchToKnowledgeBase() {
        this.state.currentView = 'knowledge_base';
    }

    _onKeyDown(ev) {
        if (ev.key === "Enter" && !ev.shiftKey) {
            ev.preventDefault();
            this.sendMessage();
        }
    }

    createNewChat() {
        this.state.currentChatId = null;
        this.state.messages = [];
        this.state.input = "";
        this.state.isTyping = false;
        this.state.currentView = 'chat';
    }

    async loadConversations() {
        try {
            this.state.conversations = await this.orm.call("ask.odoo.model", "get_sidebar_conversations");
        } catch (e) {
            console.error("Failed to load conversations", e);
        }
    }

    async selectChat(chatId) {
        this.state.currentChatId = chatId;
        this.state.messages = []; // Clear while loading
        this.state.currentView = 'chat';

        try {
            const history = await this.orm.call("ask.odoo.model", "get_messages", [chatId]);
            // Format and load history
            this.state.messages = history.map(msg => ({
                id: msg.id,
                text: this.formatMessage(msg.text),
                type: msg.type
            }));
        } catch (e) {
            console.error("Failed to load chat history", e);
        }
    }

    async loadDocuments() {
        try {
            this.state.documents = await this.orm.call("ask.odoo.knowledge.document", "get_all_documents");
        } catch (e) {
            console.error("Failed to load documents", e);
        }
    }

    triggerUpload() {
        // Trigger the hidden file input
        const fileInput = this.fileInputRef.el;
        if (fileInput) {
            fileInput.click();
        }
    }

    async onFileUpload(ev) {
        const file = ev.target.files[0];
        if (!file) return;

        // 1. Optimistic UI Update
        const tempId = `temp_${Date.now()}`;
        const tempDoc = {
            id: tempId,
            name: file.name,
            description: "Uploading and processing...",
            lastUpdated: "Just now",
            processing: true
        };

        // Add to beginning of list
        this.state.documents.unshift(tempDoc);

        const reader = new FileReader();
        reader.readAsDataURL(file);

        reader.onload = async () => {
            const base64Data = reader.result.split(',')[1];
            try {
                // 2. Call Backend (Heavy Process)
                await this.orm.call("ask.odoo.knowledge.document", "create_document", [
                    file.name,
                    base64Data,
                    file.name
                ]);

                // 3. Success: Refresh list (replaces temp doc with real one)
                await this.loadDocuments();

            } catch (e) {
                console.error("File upload failed", e);
                // 4. Error: Remove temp doc
                this.state.documents = this.state.documents.filter(d => d.id !== tempId);
                alert("Failed to upload document: " + (e.message || e));
            } finally {
                // Reset file input
                ev.target.value = "";
            }
        };
    }

    async deleteDocument(docId) {
        if (!confirm("Are you sure you want to delete this document?")) return;
        try {
            await this.orm.call("ask.odoo.knowledge.document", "delete_document", [docId]);
            await this.loadDocuments();
        } catch (e) {
            console.error("Failed to delete document", e);
        }
    }

    // Simple regex-based markdown parser
    formatMessage(text) {
        if (!text) return "";

        // 1. Escape HTML to prevent injection (since we trust our simple parser, but input could be anything)
        let safeText = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        // 2. Extract code blocks to prevent formatting inside them
        const codeBlocks = [];
        safeText = safeText.replace(/```(\w+)?\s*([\s\S]*?)```/g, (match, lang, code) => {
            codeBlocks.push({ lang: lang || 'text', code: code });
            return `__CODE_BLOCK_${codeBlocks.length - 1}__`;
        });

        // 3. Format Markdown syntax
        let formatted = safeText
            // Headers (e.g., ### Header)
            .replace(/^(#{1,6})\s+(.*)$/gm, (match, hashes, content) => {
                const level = hashes.length;
                return `<h${level} class="fw-bold mt-2 mb-1">${content}</h${level}>`;
            })
            // Bold (**text**)
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
            // Italic (*text*)
            .replace(/\*(.*?)\*/g, "<em>$1</em>")
            // Inline Code (`text`)
            .replace(/`([^`]+)`/g, "<code>$1</code>")
            // Lists (- item or * item) - Simple single-level replacement
            .replace(/^\s*[-*]\s+(.*)$/gm, "<li>$1</li>")
            // Wrap consecutive <li> into <ul> (Simple heuristic)
            .replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>")
            // Newlines to <br> (but not inside lists/headers ideally, but simple replace works closely enough for now)
            .replace(/\n/g, "<br>");

        // 4. Restore Code Blocks
        formatted = formatted.replace(/__CODE_BLOCK_(\d+)__/g, (match, index) => {
            const block = codeBlocks[index];
            return `<pre><div class="text-muted small mb-1">${block.lang}</div><code>${block.code}</code></pre>`;
        });

        return markup(formatted);
    }

    async sendMessage() {
        const text = this.state.input.trim();
        if (!text) return;

        const isNewChat = !this.state.currentChatId;

        // Add User Message (Formatted)
        this.state.messages.push({
            id: Date.now(),
            text: this.formatMessage(text), // Use markup
            type: 'user',
        });
        this.state.input = "";
        this.state.isTyping = true;
        this.scrollToBottom();

        try {
            // CALL PYTHON SERVER!
            const result = await this.orm.call(
                "ask.odoo.model",     // Model Name
                "process_message",    // Method Name
                [text, this.state.currentChatId, this.state.chatMode] // Arguments
            );

            // Update Chat ID (essential if it was null/new)
            this.state.currentChatId = result.conversation_id;

            // Add AI Response (Formatted)
            this.state.messages.push({
                id: Date.now() + 1,
                text: this.formatMessage(result.response), // Use markup
                type: 'ai',
                avatar: '🤖'
            });

            // Handle Action Confirmation (if code is returned)
            if (result.action_code) {
                this.state.messages.push({
                    id: Date.now() + 2,
                    type: 'confirmation',
                    code: result.action_code,
                    executed: false,
                    cancelled: false,
                    executing: false
                });
            }

            // Refresh sidebar if it was a new chat to show the new title
            if (isNewChat) {
                await this.loadConversations();
            }

        } catch (error) {
            console.error(error);
            this.state.messages.push({
                id: Date.now() + 1,
                text: this.formatMessage("Error: Could not connect to Odoo server. " + error.message),
                type: 'ai',
                avatar: '⚠️'
            });
        } finally {
            this.state.isTyping = false;
            this.scrollToBottom();
        }
    }

    async executeAction(msg) {
        if (msg.executed || msg.cancelled) return;

        msg.executing = true; // fast state update

        try {
            const result = await this.orm.call("ask.odoo.model", "execute_confirmed_code", [msg.code, this.state.currentChatId]);

            msg.executed = true; // Mark as done blocks buttons

            // Show Result
            this.state.messages.push({
                id: Date.now(),
                text: result.status === 'success' ? `✅ **Action Executed:**\n${result.result}` : `❌ **Error:**\n${result.message}`,
                type: 'ai', // reusing AI type for result display is fine
                avatar: '⚙️'
            });

        } catch (e) {
            console.error(e);
            alert("Execution Failed: " + e.message);
        } finally {
            msg.executing = false;
            this.scrollToBottom();
        }
    }

    cancelAction(msg) {
        msg.cancelled = true;
    }
}

registry.category("actions").add("odoo_ai_chatbot.chat_client", AiChat);
