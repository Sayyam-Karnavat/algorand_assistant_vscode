const fs = require('fs');
const path = require('path');

let questionAnswerData = [];
let tfidfMatrix = [];
let idfScores = {};
let vocabulary = new Set();
const stopWords = new Set([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
    'to', 'was', 'were', 'will', 'with'
]);

// Helper function to preprocess text
function preprocessText(text) {
    return text
        .toLowerCase()
        .replace(/[^\w\s]/g, '') // Remove punctuation
        .split(/\s+/)
        .filter(word => word.length > 0 && !stopWords.has(word));
}

// Compute term frequency for a document
function computeTF(words) {
    const tf = {};
    const wordCount = words.length;
    words.forEach(word => {
        tf[word] = (tf[word] || 0) + 1;
    });
    for (let word in tf) {
        tf[word] = tf[word] / wordCount;
    }
    return tf;
}

// Compute inverse document frequency
function computeIDF(documents) {
    const docCount = documents.length;
    const idf = {};
    const wordDocCount = {};

    documents.forEach(doc => {
        const uniqueWords = new Set(doc);
        uniqueWords.forEach(word => {
            wordDocCount[word] = (wordDocCount[word] || 0) + 1;
        });
    });

    for (let word in wordDocCount) {
        idf[word] = Math.log(docCount / (1 + wordDocCount[word]));
    }
    return idf;
}

// Compute TF-IDF vector for a document
function computeTFIDF(tf, idf) {
    const tfidf = {};
    for (let word in tf) {
        tfidf[word] = tf[word] * (idf[word] || 0);
    }
    return tfidf;
}

// Cosine similarity between two vectors
function cosineSimilarity(vec1, vec2) {
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let word in vec1) {
        if (vec2[word]) {
            dotProduct += vec1[word] * vec2[word];
        }
        norm1 += vec1[word] ** 2;
    }
    for (let word in vec2) {
        norm2 += vec2[word] ** 2;
    }
    norm1 = Math.sqrt(norm1);
    norm2 = Math.sqrt(norm2);

    return norm1 * norm2 === 0 ? 0 : dotProduct / (norm1 * norm2);
}

function loadQuestionAnswer(context) {
    try {
        const jsonPath = path.join(context, 'src', 'qa_pairs.json');
        const data = fs.readFileSync(jsonPath, 'utf8');
        questionAnswerData = JSON.parse(data);

        // Preprocess questions and build vocabulary
        const documents = questionAnswerData.map(qa => preprocessText(qa.question));
        documents.forEach(doc => {
            doc.forEach(word => vocabulary.add(word));
        });

        // Compute TF-IDF for each question
        idfScores = computeIDF(documents);
        tfidfMatrix = documents.map(doc => {
            const tf = computeTF(doc);
            return computeTFIDF(tf, idfScores);
        });

    } catch (error) {
        throw new Error(`Error loading qa_pairs.json: ${error.message}`);
    }
}

function query(text) {
    try {
        // Preprocess query
        const queryWords = preprocessText(text);
        const queryTF = computeTF(queryWords);
        const queryTFIDF = computeTFIDF(queryTF, idfScores);

        let bestScore = 0;
        let bestAnswer = 'No relevant answer found.';
        tfidfMatrix.forEach((docTFIDF, index) => {
            const score = cosineSimilarity(queryTFIDF, docTFIDF);
            if (score > bestScore && score > 0.1) { // Threshold to avoid irrelevant matches
                bestScore = score;
                bestAnswer = questionAnswerData[index].answer;
            }
        });

        return bestAnswer;
    } catch (error) {
        return `Error processing query: ${error.message}`;
    }
}

module.exports = { loadQuestionAnswer, query };