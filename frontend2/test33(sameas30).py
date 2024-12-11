import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from functools import lru_cache
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import islice
import numpy as np

class OptimizedChatAnalyzer:
    def __init__(self, api_endpoint: str = None, api_key: str = None, batch_size: int = 10):
        """Initialize the analyzer with optional LLM support and optimization parameters"""
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        } if api_key else None
        
        self.llm_available = bool(api_endpoint and api_key)
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.sectors = self._get_default_sectors()  # Initialize with default sectors
        self.last_update = None
        self.batch_size = batch_size
        self.sentiment_cache = {}
        self.category_cache = {}
        
        # Initialize thread pool for I/O operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a text"""
        return self.sentence_model.encode([text])[0]

    def _compute_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Compute embeddings for a batch of texts"""
        return [self._compute_embedding(text) for text in texts]

    def _compute_similarities(self, text_embedding: np.ndarray, sector_embeddings: List[np.ndarray]) -> List[float]:
        """Compute cosine similarity"""
        return list(cosine_similarity([text_embedding], sector_embeddings)[0])

    async def _analyze_sentiment_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for a batch of texts concurrently"""
        if not self.llm_available:
            return [{"sentiment": "neutral", "confidence": 0.5, "reasoning": "LLM not available"}] * len(texts)

        async def process_single(text):
            cache_key = hash(text)
            if cache_key in self.sentiment_cache:
                return self.sentiment_cache[cache_key]

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.api_endpoint, headers=self.headers, json={
                        "messages": [
                            {"role": "system", "content": """Analyze the workplace message sentiment.
                            Return ONLY a JSON object with:
                            - sentiment: "positive", "negative", "neutral", or "mixed"
                            - confidence: float between 0 and 1
                            - reasoning: brief explanation"""},
                            {"role": "user", "content": f"Analyze: '{text}'"}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 150
                    }) as response:
                        if response.status == 200:
                            result = await response.json()
                            sentiment_result = json.loads(result['choices'][0]['message']['content'])
                            self.sentiment_cache[cache_key] = sentiment_result
                            return sentiment_result
            except Exception as e:
                print(f"Error in sentiment analysis: {str(e)}")
            return {"sentiment": "neutral", "confidence": 0.5, "reasoning": "Error in analysis"}

        tasks = [process_single(text) for text in texts]
        return await asyncio.gather(*tasks)

    async def _get_message_categories_batch(self, texts: List[str]) -> List[str]:
        """Get message categories for a batch of texts"""
        # Compute embeddings
        text_embeddings = self._compute_embeddings_batch(texts)
        sector_embeddings = [self._compute_embedding(sector["name"]) for sector in self.sectors]

        # Compute similarities
        categories = []
        for text_embedding in text_embeddings:
            similarities = self._compute_similarities(text_embedding, sector_embeddings)
            max_idx = np.argmax(similarities)
            categories.append(self.sectors[max_idx]["name"])

        return categories

    async def analyze_responses_by_period(self, messages: List[Dict[str, Any]], period_days: Optional[int] = None) -> Dict[str, Any]:
        """Analyze responses for a specific time period using batch processing"""
        period_messages = self._filter_messages_by_period(messages, period_days)
        
        results = {
            "summary": {
                "total_messages": len(period_messages),
                "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0},
                "sector_distribution": {},
                "average_sentiment": 0.0,
                "period_days": period_days,
                "summary_text": None
            },
            "categories": {}
        }

        if not period_messages:
            return results

        total_sentiment_score = 0.0

        # Process messages in batches
        for batch in self._batch_iterator(period_messages, self.batch_size):
            batch_texts = [msg["content"] for msg in batch]
            
            try:
                # Parallel sentiment analysis
                sentiment_results = await self._analyze_sentiment_batch(batch_texts)
                
                # Get categories
                categories = await self._get_message_categories_batch(batch_texts)
                
                # Update results
                for msg, sentiment, category in zip(batch, sentiment_results, categories):
                    if category not in results["categories"]:
                        results["categories"][category] = {
                            "messages": [],
                            "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
                        }
                    
                    results["categories"][category]["messages"].append({
                        "content": msg["content"],
                        "sentiment": sentiment["sentiment"],
                        "confidence": sentiment["confidence"],
                        "reasoning": sentiment["reasoning"]
                    })
                    
                    results["categories"][category]["sentiment_distribution"][sentiment["sentiment"]] += 1
                    results["summary"]["sentiment_distribution"][sentiment["sentiment"]] += 1
                    results["summary"]["sector_distribution"][category] = \
                        results["summary"]["sector_distribution"].get(category, 0) + 1
                    
                    # Update sentiment score
                    sentiment_value = 1 if sentiment["sentiment"] == "positive" else -1 if sentiment["sentiment"] == "negative" else 0
                    total_sentiment_score += sentiment["confidence"] * sentiment_value

            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue

        # Calculate average sentiment
        if results["summary"]["total_messages"] > 0:
            results["summary"]["average_sentiment"] = total_sentiment_score / results["summary"]["total_messages"]

        # Generate summaries
        try:
            for category, data in results["categories"].items():
                summary = await self._generate_category_summary(data["messages"])
                results["categories"][category]["summary"] = summary

            results["summary"]["summary_text"] = await self._generate_overall_summary(
                results["categories"],
                results["summary"]["sentiment_distribution"]
            )
        except Exception as e:
            print(f"Error generating summaries: {str(e)}")

        return results

    @staticmethod
    def _batch_iterator(iterable, batch_size):
        """Helper function to create batches from an iterable"""
        iterator = iter(iterable)
        while batch := list(islice(iterator, batch_size)):
            yield batch

    async def _generate_category_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Generate a summary for a category's messages using LLM"""
        if not self.llm_available or not messages:
            return "Summary not available"

        messages_text = "\n".join([f"- {msg['content']} (Sentiment: {msg['sentiment']})" for msg in messages])
        
        prompt = {
            "messages": [
                {"role": "system", "content": """Analyze the workplace messages and provide a concise summary.
                Focus on key themes, patterns, and overall sentiment. Keep the summary brief but informative."""},
                {"role": "user", "content": f"Summarize these workplace messages:\n{messages_text}"}
            ],
            "temperature": 0.3,
            "max_tokens": 150
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_endpoint, headers=self.headers, json=prompt) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content'].strip()
                    else:
                        return "Failed to generate summary"
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    async def _generate_overall_summary(self, category_analysis: Dict[str, Any], overall_sentiment: Dict[str, int]) -> str:
        """Generate an overall summary for the time period using LLM"""
        if not self.llm_available:
            return "Overall summary not available"

        summary_data = {
            "categories": category_analysis,
            "overall_sentiment": overall_sentiment
        }
        
        prompt = {
            "messages": [
                {"role": "system", "content": """Generate a concise overall summary of workplace feedback.
                Consider all categories and their sentiments. Focus on key trends and important insights.
                Keep the summary clear and actionable."""},
                {"role": "user", "content": f"Provide an overall summary based on this analysis:\n{json.dumps(summary_data, indent=2)}"}
            ],
            "temperature": 0.3,
            "max_tokens": 200
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_endpoint, headers=self.headers, json=prompt) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content'].strip()
                    else:
                        return "Failed to generate overall summary"
        except Exception as e:
            return f"Error generating overall summary: {str(e)}"

    def _filter_messages_by_period(self, messages: List[Dict[str, Any]], period_days: Optional[int] = None) -> List[Dict[str, Any]]:
        """Filter messages based on time period"""
        if not period_days:  # For overall analysis
            return messages.get("responses", []) if isinstance(messages, dict) else messages
            
        try:
            # Handle both direct list of messages and nested responses
            msg_list = messages.get("responses", []) if isinstance(messages, dict) else messages
            cutoff_date = datetime.now() - timedelta(days=period_days)
            
            filtered_messages = []
            for msg in msg_list:
                try:
                    # Get timestamp with fallback
                    timestamp_str = msg.get("timestamp")
                    if not timestamp_str:
                        continue
                        
                    # Parse timestamp and compare
                    msg_date = datetime.fromisoformat(timestamp_str).replace(tzinfo=None)
                    if msg_date >= cutoff_date:
                        filtered_messages.append(msg)
                except (ValueError, AttributeError) as e:
                    print(f"Error parsing timestamp for message: {e}")
                    continue
                    
            return filtered_messages
            
        except Exception as e:
            print(f"Error filtering messages by period: {e}")
            return []

    def _get_period_name(self, period_days: Optional[int]) -> str:
        """Get human-readable name for time period"""
        if period_days is None:
            return "Overall"
        period_mapping = {
            7: "Last Week",
            30: "Last Month",
            90: "Last 3 Months",
            180: "Last 6 Months",
            365: "Last Year"
        }
        return period_mapping.get(period_days, f"Last {period_days} days")

    @staticmethod
    def _get_default_sectors() -> List[Dict[str, Any]]:
        """Fallback default sectors if LLM is unavailable"""
        return [
            {
                "name": "Assess Work Environment and Demographics",
                "keywords": ["environment", "workplace", "office", "demographics", "diversity"],
                "patterns": ["work environment", "office space", "diversity"],
                "description": "Work environment and demographic analysis"
            },
            {
                "name": "Evaluate Physical Health and Lifestyle",
                "keywords": ["health", "physical", "exercise", "lifestyle"],
                "patterns": ["physical health", "lifestyle", "exercise"],
                "description": "Physical health and lifestyle assessment"
            },
            {
                "name": "Mental and Emotional Well-being",
                "keywords": ["mental", "emotional", "wellbeing", "stress"],
                "patterns": ["mental health", "emotional wellbeing", "stress"],
                "description": "Mental and emotional health analysis"
            },
            {
                "name": "Safety Culture and Work Pressure",
                "keywords": ["safety", "pressure", "workload", "stress"],
                "patterns": ["work pressure", "safety", "workload"],
                "description": "Safety and work pressure analysis"
            },
            {
                "name": "Work-Life Balance and Support Services Awareness",
                "keywords": ["balance", "support", "services", "life"],
                "patterns": ["work life balance", "support services"],
                "description": "Work-life balance and support services"
            },
            {
                "name": "Resilience and Adaptability",
                "keywords": ["resilience", "adaptability", "change", "growth"],
                "patterns": ["resilience", "adaptability", "change"],
                "description": "Resilience and adaptability assessment"
            },
            {
                "name": "Technology Integration",
                "keywords": ["technology", "tools", "software", "digital"],
                "patterns": ["technology", "digital tools", "software"],
                "description": "Technology integration and usage"
            }
        ]

    async def refresh_sectors(self, messages: List[Dict[str, str]] = None):
        """Refresh sectors based on recent messages"""
        if not self.llm_available:
            self.sectors = self._get_default_sectors()
            return

        prompt = {
            "messages": [
                {"role": "system", "content": """You are a JSON-only response API for workplace analysis.
                Generate a list of workplace sectors. Your response must be ONLY a valid JSON array of sector objects.
                Each sector object must have:
                - name: string
                - keywords: array of strings
                - patterns: array of regex patterns (use raw strings, no escaping needed)
                - description: string"""},
                {"role": "user", "content": f"Current sectors: {json.dumps(self.sectors)}\nRecent messages: {json.dumps(messages) if messages else 'No messages provided'}"}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_endpoint, headers=self.headers, json=prompt) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        try:
                            updated_sectors = json.loads(content)
                            if isinstance(updated_sectors, list) and updated_sectors:
                                self.sectors = updated_sectors
                                self.last_update = datetime.now()
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse LLM response: {e}")
                            self.sectors = self._get_default_sectors()
                    else:
                        print(f"LLM API error: {response.status}")
                        self.sectors = self._get_default_sectors()
        except Exception as e:
            print(f"Error refreshing sectors: {e}")
            self.sectors = self._get_default_sectors()

def save_to_mock_data(all_results):
    """Save analysis results to mockAnalysisData.js"""
    output = []
    
    # We'll focus on week and month analysis as per the requested format
    periods_to_show = ["Last Week", "Last Month"]
    
    for period in periods_to_show:
        if period not in all_results:
            continue
            
        results = all_results[period]
        
        output.append(f"=== Analysis for {period} ===")
        output.append(f"Total Messages: {results['summary']['total_messages']}")
        output.append(f"Overall Sentiment Distribution: {results['summary']['sentiment_distribution']}")
        output.append(f"Average Sentiment: {results['summary']['average_sentiment']:.2f}\n")
        
        output.append("Category-wise Analysis:\n")
        
        for category, data in results["categories"].items():
            output.append(f"{category}:")
            output.append(f"Messages: {len(data['messages'])}")
            output.append(f"Sentiment Distribution: {data['sentiment_distribution']}")
            output.append(f"Summary: {data.get('summary', 'No summary available')}\n")
        
        output.append("Overall Period Summary:")
        if "summary_text" in results["summary"]:
            output.append(f"{results['summary']['summary_text']}\n")
        else:
            output.append("No summary available\n")
    
    # Create the JavaScript file content
    js_content = 'const mockData = `\n'
    js_content += ''.join(f"{line}\n" for line in output)
    js_content += '`;\n\nexport default mockData;'

    # Save to file
    output_path = "mockAnalysisData.js"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(js_content)
    print(f"Analysis data saved to {output_path}")

async def main():
    # Example usage
    api_endpoint = "https://dev-test-2.openai.azure.com/openai/deployments/Gpt4omini_test/chat/completions?api-version=2024-02-15-preview"
    api_key = "7535acae5fe648208168aa9e5f29f83b"
    
    # Create analyzer with batch processing
    analyzer = OptimizedChatAnalyzer(api_endpoint, api_key, batch_size=10)
    await analyzer.refresh_sectors()
    
    # Load and analyze messages
    with open("responses.json", "r") as f:
        data = json.load(f)
    
    # Filter responses for TechCorp
    messages = []
    for response in data["responses"]:
        if response["company"].lower() == "techcorp":
            messages.append({
                "timestamp": response["timestamp"],
                "content": response["content"]
            })
    
    # Define time periods for analysis
    time_periods = [7, 30, 90, 180, 365, None]  # days (None for all-time)
    
    # Analyze each time period
    all_results = {}
    for period in time_periods:
        results = await analyzer.analyze_responses_by_period(messages, period)
        period_name = analyzer._get_period_name(period)
        all_results[period_name] = results
        
        # Print results
        print(f"\n=== Analysis for {period_name} ===")
        print(f"Total Messages: {results['summary']['total_messages']}")
        print(f"Overall Sentiment Distribution: {results['summary']['sentiment_distribution']}")
        print(f"Average Sentiment: {results['summary']['average_sentiment']:.2f}\n")
        
        print("Category-wise Analysis:\n")
        for category, data in results["categories"].items():
            print(f"{category}:")
            print(f"Messages: {len(data['messages'])}")
            print(f"Sentiment Distribution: {data['sentiment_distribution']}")
            print(f"Summary: {data.get('summary', 'No summary available')}\n")
        
        print("Overall Period Summary:")
        if "summary_text" in results["summary"]:
            print(results["summary"]["summary_text"])
        else:
            print("No summary available")
    
    # Save results to mockAnalysisData.js
    save_to_mock_data(all_results)

if __name__ == "__main__":
    asyncio.run(main())
    # Run datatransform.py after completing the analysis
    import subprocess
    import os
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datatransform_path = os.path.join(current_dir, "datatransform.py")
    
    # Run datatransform.py using Python interpreter
    subprocess.run(["python", datatransform_path], check=True)
    