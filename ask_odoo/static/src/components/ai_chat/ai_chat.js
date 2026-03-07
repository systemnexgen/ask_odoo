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
        this.notification = useService("notification");
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
            openMenuChatId: null,
            isListening: false,
        });
        this.messagesEndRef = useRef("messagesEnd");
        this.fileInputRef = useRef("fileInput");
        this._speechRecognition = null;
        this._chartInstances = {};

        onWillStart(async () => {
            await this.loadConversations();
            await this.loadDocuments();
        });

        useEffect(() => {
            this.scrollToBottom();
            // Render any pending charts after DOM update
            this._renderPendingCharts();
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

        // Notify user that refresh has started
        this.notification.add("Schema refresh started. This may take a few moments...", {
            type: "info",
            title: "Processing",
        });

        try {
            await this.orm.call("ask.odoo.model", "refresh_schema_index");
            this.notification.add("Schema Refreshed Successfully!", {
                type: "success",
                title: "Success",
            });
        } catch (e) {
            console.error(e);
            this.notification.add("Failed to refresh schema: " + e.message, {
                type: "danger",
                title: "Error",
            });
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
        this.state.openMenuChatId = null;
    }

    toggleChatMenu(chatId) {
        this.state.openMenuChatId = this.state.openMenuChatId === chatId ? null : chatId;
    }

    async deleteConversation(chatId) {
        this.state.openMenuChatId = null;
        try {
            await this.orm.call("ask.odoo.model", "delete_conversation", [chatId]);
            // If the deleted chat was active, clear the view
            if (this.state.currentChatId === chatId) {
                this.state.currentChatId = null;
                this.state.messages = [];
            }
            await this.loadConversations();
        } catch (e) {
            console.error("Failed to delete conversation", e);
            this.notification.add("Failed to delete conversation: " + e.message, {
                type: "danger",
                title: "Error",
            });
        }
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
        this.state.openMenuChatId = null;

        try {
            const history = await this.orm.call("ask.odoo.model", "get_messages", [chatId]);
            const loadedMessages = [];

            for (const msg of history) {
                if (msg.type === 'confirmation') {
                    // Restore confirmation card (already executed since it's from history)
                    loadedMessages.push({
                        id: msg.id,
                        type: 'confirmation',
                        code: msg.action_code || '',
                        executed: true,
                        cancelled: false,
                        executing: false,
                    });
                } else if (msg.type === 'chart') {
                    // This shouldn't happen since we store chart_data on 'ai' messages,
                    // but handle it just in case
                    if (msg.chart_data) {
                        const chartId = `chart_hist_${msg.id}`;
                        loadedMessages.push({
                            id: msg.id,
                            type: 'chart',
                            chartId: chartId,
                            chartData: msg.chart_data,
                        });
                    }
                } else {
                    // Standard user/ai message
                    // If it has chart_data, inject a chart message BEFORE the table
                    if (msg.chart_data) {
                        const chartId = `chart_hist_${msg.id}`;
                        loadedMessages.push({
                            id: msg.id * 1000,
                            type: 'chart',
                            chartId: chartId,
                            chartData: msg.chart_data,
                        });
                    }

                    // If it has result_html, render that instead of plain text
                    if (msg.result_html) {
                        loadedMessages.push({
                            id: msg.id,
                            text: markup(`<div class="text-start w-100">
                                <strong>\u2705 Action Executed:</strong>
                                <div class="table-responsive dataframe-wrapper mt-2">${msg.result_html}</div>
                            </div>`),
                            type: 'ai',
                        });
                    } else {
                        loadedMessages.push({
                            id: msg.id,
                            text: this.formatMessage(msg.text),
                            type: msg.type,
                        });
                    }
                }
            }

            this.state.messages = loadedMessages;
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
                this.notification.add("Failed to upload document: " + (e.message || e), {
                    type: "danger",
                    title: "Upload Failed",
                });
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

        // A. Tables (Simple GFM style)
        // Look for headers: | ... | ... | followed by separator: | --- | --- |
        safeText = safeText.replace(/(\|[^\n]+\|\r?\n)((?:\|:?[-]+:?)+\|)(\r?\n(?:\|[^\n]+\|\r?\n?)*)/g, (match, header, separator, body) => {
            try {
                const parseRow = (row) => row.trim().split('|').filter(c => c && c.trim() !== '').map(c => c.trim());

                const headers = parseRow(header);
                const separators = parseRow(separator);
                const alignments = separators.map(s => {
                    if (s.startsWith(':') && s.endsWith(':')) return 'center';
                    if (s.endsWith(':')) return 'right';
                    return 'left';
                });

                let html = '<div class="overflow-auto mb-3"><table class="dataframe">'; // Use .dataframe to match our SCSS

                // Head
                html += '<thead><tr>';
                headers.forEach((h, i) => {
                    const align = alignments[i] || 'left';
                    html += `<th style="text-align: ${align}">${h}</th>`;
                });
                html += '</tr></thead>';

                // Body
                html += '<tbody>';
                const rows = body.trim().split('\n');
                rows.forEach(row => {
                    // Skip empty rows
                    if (!row.trim()) return;

                    const cells = parseRow(row);
                    if (cells.length === 0) return;

                    html += '<tr>';
                    cells.forEach((c, i) => {
                        const align = alignments[i] || 'left';
                        // Handle empty cells
                        const content = c || '';
                        html += `<td style="text-align: ${align}">${content}</td>`;
                    });
                    html += '</tr>';
                });
                html += '</tbody></table></div>';

                return html;
            } catch (e) {
                console.error("Table parsing failed", e);
                return match; // Fallback to raw text
            }
        });

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
            // Newlines to <br> (but not inside lists/headers/tables)
            // We use a negative lookbehind/lookahead or just keep it simple.
            // Since tables use <div> wrapper, we want to respect that.
            // Simplified: we only replace newlines that are NOT inside tags we just parsed (hX, table, ul, li)
            // But strict regex for that is hard. 
            // Simple approach: Replace \n with <br> ONLY if it's not followed by a block tag?
            // Actually, for now, normal text newlines -> br is fine.
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
            // Show Result
            let messageContent;
            if (result.status === 'success') {
                const isHtml = typeof result.result === 'string' && (result.result.trim().startsWith('<table') || result.result.trim().startsWith('<div'));

                // If chart data is available, push a chart message FIRST (above the table)
                if (result.chart_data) {
                    const chartId = `chart_${Date.now()}`;
                    this.state.messages.push({
                        id: Date.now(),
                        type: 'chart',
                        chartId: chartId,
                        chartData: result.chart_data,
                    });
                }

                if (isHtml) {
                    messageContent = markup(`<div class="text-start w-100">
                        <strong>✅ Action Executed:</strong>
                        <div class="table-responsive dataframe-wrapper mt-2">${result.result}</div>
                    </div>`);
                } else {
                    messageContent = this.formatMessage(`✅ **Action Executed:**\n${result.result}`);
                }
            } else {
                messageContent = this.formatMessage(result.message);
            }

            this.state.messages.push({
                id: Date.now() + 1,
                text: messageContent,
                type: 'ai',
                avatar: '⚙️'
            });

            // If the LLM proposed a corrected code fix, add a new confirmation card
            // so the user can approve the retry with one click.
            // Hard cap: after 3 user-visible retries, stop and ask them to rephrase.
            if (result.status === 'error' && result.retry_code) {
                const currentDepth = msg.retryDepth || 0;
                if (currentDepth < 3) {
                    this.state.messages.push({
                        id: Date.now() + 2,
                        type: 'confirmation',
                        code: result.retry_code,
                        executed: false,
                        cancelled: false,
                        executing: false,
                        isRetry: true,
                        retryDepth: currentDepth + 1,
                    });
                } else {
                    // Terminal failure — do not loop further
                    this.state.messages.push({
                        id: Date.now() + 2,
                        text: this.formatMessage(
                            "⛔ **Unable to automatically fix this query after 3 attempts.**\n\n" +
                            "The model I tried to access may not be installed on your Odoo instance, " +
                            "or the question may need more specific details.\n\n" +
                            "**Please try rephrasing your question**, for example:\n" +
                            "- Mention the exact module or record type\n" +
                            "- Check that the relevant Odoo app is installed"
                        ),
                        type: 'ai',
                        avatar: '⛔',
                    });
                }
            }

        } catch (e) {
            console.error(e);
            this.notification.add("Execution Failed: " + e.message, {
                type: "danger",
                title: "Execution Error",
            });
        } finally {
            msg.executing = false;
            this.scrollToBottom();
        }
    }

    cancelAction(msg) {
        msg.cancelled = true;
    }

    _renderPendingCharts() {
        // Find all chart messages that haven't been rendered yet
        for (const msg of this.state.messages) {
            if (msg.type !== 'chart' || msg._rendered) continue;

            const canvas = document.getElementById(msg.chartId);
            if (!canvas) continue;

            // Destroy previous instance if it exists
            if (this._chartInstances[msg.chartId]) {
                this._chartInstances[msg.chartId].destroy();
            }

            const chartData = msg.chartData;
            const isDoughnut = chartData.type === 'doughnut';

            try {
                this._chartInstances[msg.chartId] = new Chart(canvas, {
                    type: chartData.type,
                    data: {
                        labels: chartData.labels,
                        datasets: chartData.datasets,
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: true,
                        plugins: {
                            title: {
                                display: true,
                                text: chartData.title || '',
                                font: { size: 14, weight: 'bold' },
                                padding: { bottom: 16 },
                            },
                            legend: {
                                display: chartData.datasets.length > 1 || isDoughnut,
                                position: isDoughnut ? 'right' : 'top',
                                labels: { usePointStyle: true, padding: 16 },
                            },
                            tooltip: {
                                backgroundColor: 'rgba(0,0,0,0.8)',
                                cornerRadius: 8,
                                padding: 12,
                            },
                        },
                        scales: isDoughnut ? {} : {
                            x: {
                                grid: { display: false },
                                ticks: { maxRotation: 45, autoSkip: true },
                            },
                            y: {
                                beginAtZero: true,
                                grid: { color: 'rgba(0,0,0,0.06)' },
                            },
                        },
                        animation: {
                            duration: 800,
                            easing: 'easeOutQuart',
                        },
                    },
                });
                msg._rendered = true;
            } catch (e) {
                console.error('Chart rendering failed:', e);
            }
        }
    }

    toggleVoiceInput() {
        if (this.state.isListening) {
            this.stopVoiceInput();
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            this.notification.add(
                "Voice input is not supported by your browser. Please use Chrome or Edge.",
                { type: "warning", title: "Unsupported Browser" }
            );
            return;
        }

        const recognition = new SpeechRecognition();
        recognition.lang = "en-US";
        recognition.interimResults = true;
        recognition.continuous = true;
        recognition.maxAlternatives = 1;

        // Track text that existed before we started recording
        const existingText = this.state.input;

        recognition.onresult = (event) => {
            let interimTranscript = "";
            let finalTranscript = "";

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }

            // Build the full input: existing text + final (committed) + interim (live preview)
            const separator = existingText ? " " : "";
            this.state.input = existingText + separator + finalTranscript + interimTranscript;
        };

        recognition.onerror = (event) => {
            console.error("Speech recognition error:", event.error);
            if (event.error !== "aborted") {
                this.notification.add(
                    "Voice input error: " + event.error,
                    { type: "danger", title: "Microphone Error" }
                );
            }
            this.state.isListening = false;
            this._speechRecognition = null;
        };

        recognition.onend = () => {
            this.state.isListening = false;
            this._speechRecognition = null;
        };

        recognition.start();
        this._speechRecognition = recognition;
        this.state.isListening = true;
    }

    stopVoiceInput() {
        if (this._speechRecognition) {
            this._speechRecognition.stop();
            this._speechRecognition = null;
        }
        this.state.isListening = false;
    }
}

registry.category("actions").add("odoo_ai_chatbot.chat_client", AiChat);
