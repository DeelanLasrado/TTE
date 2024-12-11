// Mock data to simulate API response
const mockData = {
    "Last Week": {
        totalMessages: 167,
        overallSentimentDistribution: { positive: 95, negative: 15, neutral: 12, mixed: 45 },
        averageSentiment: 0.44,
        categories: {
            "Work-Life Balance and Support Services Awareness": {
                messages: 116,
                sentimentDistribution: { positive: 76, negative: 11, neutral: 4, mixed: 25 },
                summary: "The workplace messages reflect a complex landscape of sentiments, primarily revolving around themes of recognition, management style, work-life balance, and team dynamics."
            },
            "Safety Culture and Work Pressure": {
                messages: 27,
                sentimentDistribution: { positive: 10, negative: 2, neutral: 7, mixed: 8 },
                summary: "The workplace messages reflect a complex environment characterized by a mix of positive, neutral, and negative sentiments. Key themes include safety protocols and work pressure management."
            },
            "Technology Integration": {
                messages: 11,
                sentimentDistribution: { positive: 8, negative: 0, neutral: 1, mixed: 2 },
                summary: "The workplace messages reflect a strong emphasis on innovation and the encouragement of new ideas across the company. Key themes include adoption of new technologies and digital transformation initiatives."
            },
            "Assess Work Environment and Demographics": {
                messages: 13,
                sentimentDistribution: { positive: 1, negative: 2, neutral: 0, mixed: 10 },
                summary: "The workplace messages reflect a mixed sentiment regarding inclusivity and growth opportunities within the organization. Key themes include diversity initiatives and career development programs."
            }
        },
        overallSummary: "The analysis of workplace feedback reveals a complex landscape of sentiments, predominantly positive but with significant areas for improvement. Employee engagement, work-life balance, and technological advancements are key focus areas."
    },
    "Last Month": {
        totalMessages: 720,
        overallSentimentDistribution: { positive: 410, negative: 85, neutral: 95, mixed: 130 },
        averageSentiment: 0.39,
        categories: {
            "Work-Life Balance and Support Services Awareness": {
                messages: 280,
                sentimentDistribution: { positive: 180, negative: 30, neutral: 20, mixed: 50 },
                summary: "Over the past month, employees have expressed a range of sentiments regarding work-life balance and support services. While many appreciate the existing initiatives, there's a call for more flexible working arrangements and improved mental health support."
            },
            "Safety Culture and Work Pressure": {
                messages: 220,
                sentimentDistribution: { positive: 120, negative: 35, neutral: 40, mixed: 25 },
                summary: "Safety protocols are generally well-received, but there are concerns about increasing work pressure affecting overall workplace safety. Employees suggest regular safety training refreshers and workload management strategies."
            },
            "Technology Integration": {
                messages: 130,
                sentimentDistribution: { positive: 80, negative: 10, neutral: 25, mixed: 15 },
                summary: "There's enthusiasm about new technology implementations, with employees highlighting improved efficiency. However, some express a need for more comprehensive training programs to fully leverage these tools."
            },
            "Assess Work Environment and Demographics": {
                messages: 90,
                sentimentDistribution: { positive: 30, negative: 10, neutral: 10, mixed: 40 },
                summary: "Diversity and inclusion initiatives are appreciated, but employees seek more transparent career progression paths and mentorship programs. There's a call for more diverse representation in leadership roles."
            }
        },
        overallSummary: "The month-long analysis reveals a generally positive workplace sentiment with areas for improvement. Key focus areas include enhancing work-life balance initiatives, managing work pressure, continued technology integration with adequate training, and furthering diversity and inclusion efforts."
    }
};

// DOM elements
const periodSelect = document.getElementById('period-select');
const categorySelect = document.getElementById('category-select');
const totalMessages = document.getElementById('total-messages');
const averageSentiment = document.getElementById('average-sentiment');
const sentimentDistribution = document.getElementById('sentiment-distribution');
const summaryTitle = document.getElementById('summary-title');
const summaryContent = document.getElementById('summary-content');

// Populate period select
Object.keys(mockData).forEach(period => {
    const option = document.createElement('option');
    option.value = period;
    option.textContent = period;
    periodSelect.appendChild(option);
});

// Initialize with first period
updateDashboard(Object.keys(mockData)[0], 'Overall');

// Event listeners
periodSelect.addEventListener('change', (e) => updateDashboard(e.target.value, categorySelect.value));
categorySelect.addEventListener('change', (e) => updateDashboard(periodSelect.value, e.target.value));

function updateDashboard(period, category) {
    const data = mockData[period];
    
    // Update category select
    categorySelect.innerHTML = '<option value="Overall">Overall</option>';
    Object.keys(data.categories).forEach(cat => {
        const option = document.createElement('option');
        option.value = cat;
        option.textContent = cat;
        categorySelect.appendChild(option);
    });
    categorySelect.value = category;
    
    if (category === 'Overall') {
        // Update total messages and average sentiment
        totalMessages.textContent = data.totalMessages;
        averageSentiment.textContent = data.averageSentiment.toFixed(2);
        
        // Update sentiment distribution
        updateSentimentDistribution(data.overallSentimentDistribution);
        
        // Update summary
        summaryTitle.textContent = 'Overall Summary';
        summaryContent.textContent = data.overallSummary;
    } else {
        const categoryData = data.categories[category];
        
        // Update total messages
        totalMessages.textContent = categoryData.messages;
        
        // Calculate and update average sentiment for the category
        const totalSentiment = Object.entries(categoryData.sentimentDistribution).reduce((acc, [sentiment, count]) => {
            const sentimentValue = sentiment === 'positive' ? 1 : sentiment === 'negative' ? -1 : 0;
            return acc + (sentimentValue * count);
        }, 0);
        const averageCategorySentiment = totalSentiment / categoryData.messages;
        averageSentiment.textContent = averageCategorySentiment.toFixed(2);
        
        // Update sentiment distribution
        updateSentimentDistribution(categoryData.sentimentDistribution);
        
        // Update summary
        summaryTitle.textContent = `${category} Summary`;
        summaryContent.textContent = categoryData.summary;
    }
}

function updateSentimentDistribution(distribution) {
    sentimentDistribution.innerHTML = '';
    Object.entries(distribution).forEach(([sentiment, count]) => {
        const item = document.createElement('div');
        item.className = 'sentiment-item';
        item.innerHTML = `
            <div class="sentiment-label">${sentiment}</div>
            <div class="sentiment-value">${count}</div>
        `;
        sentimentDistribution.appendChild(item);
    });
}

