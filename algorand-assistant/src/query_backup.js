const fs = require('fs');
const path = require('path');
const fuzzball = require('fuzzball');

let questionAnswerData = [];

function loadQuestionAnswer(context) {
    try {
        const jsonPath = path.join(context, 'src', 'qa_pairs.json');
        const data = fs.readFileSync(jsonPath, 'utf8');
        questionAnswerData = JSON.parse(data);
    } catch (error) {
        throw new Error(`Error loading qa_pairs.json: ${error.message}`);
    }
}

function query(text) {
    try {
        let bestScore = 0;
        let bestAnswer = 'No relevant answer found.';
        for (const qa of questionAnswerData) {
            const score = fuzzball.partial_ratio(text.toLowerCase(), qa.question.toLowerCase());
            if (score > bestScore && score > 80) {
                bestScore = score;
                bestAnswer = qa.answer;
            }
        }
        return bestAnswer;
    } catch (error) {
        return `Error processing query: ${error.message}`;
    }
}

module.exports = { loadQuestionAnswer, query };