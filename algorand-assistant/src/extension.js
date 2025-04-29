const vscode = require('vscode');
const { loadQuestionAnswer, query } = require('./query');
const fs = require('fs');
const path = require('path');

function activate(context) {
    // Load question-answer data at startup
    try {
        loadQuestionAnswer(context.extensionPath);
    } catch (error) {
        vscode.window.showErrorMessage(error.message);
        return;
    }

    // Register existing query command
    let queryDisposable = vscode.commands.registerCommand('algorandQueryAssistant.query', async () => {
        const input = await vscode.window.showInputBox({
            placeHolder: 'Enter your Algorand blockchain question',
            prompt: 'Ask a question about Algorand'
        });

        if (!input) {
            vscode.window.showWarningMessage('No query provided.');
            return;
        }

        const result = query(input);
        if (result.startsWith('Error')) {
            vscode.window.showErrorMessage(result);
        } else {
            vscode.window.showInformationMessage('Algorand Query Result', result);
        }
    });

    // Register chatbot command
    let chatbotDisposable = vscode.commands.registerCommand('algorandQueryAssistant.openChatbot', () => {
        // Create webview panel
        const panel = vscode.window.createWebviewPanel(
            'algorandChatbot', // Identifier
            'Algorand Query Chatbot', // Title
            vscode.ViewColumn.One, // Show in editor column 1
            { enableScripts: true } // Enable JavaScript in webview
        );

        // Load HTML content
        const htmlPath = path.join(context.extensionPath, 'src', 'chatbot.html');
        try {
            panel.webview.html = fs.readFileSync(htmlPath, 'utf8');
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to load chatbot UI: ${error.message}`);
            return;
        }

        // Handle messages from webview
        panel.webview.onDidReceiveMessage(
            message => {
                if (message.command === 'query') {
                    const result = query(message.text);
                    panel.webview.postMessage({ command: 'answer', text: result });
                }
            },
            undefined,
            context.subscriptions
        );
    });

    context.subscriptions.push(queryDisposable, chatbotDisposable);
}

function deactivate() {}

module.exports = { activate, deactivate };