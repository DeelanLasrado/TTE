import re
import json
import ast

def transform_analysis_data(raw_data):
    # Split the data into sections using a more robust pattern
    sections = re.split(r'===\s*Analysis for', raw_data)
    sections = [s for s in sections if s.strip()]  # Remove empty sections
    
    result = {}
    
    for section in sections:
        try:
            # Extract period (Last Week/Last Month) with a simpler pattern
            period_match = re.match(r'\s*(Last\s+\w+)\s*===', section)
            if not period_match:
                continue
            period = period_match.group(1)
            
            # Extract total messages with a simpler pattern
            total_messages_match = re.search(r'Total Messages:\s*(\d+)', section)
            total_messages = int(total_messages_match.group(1)) if total_messages_match else 0
            
            # Extract sentiment distribution with a more precise pattern
            sentiment_match = re.search(r'Overall Sentiment Distribution:\s*(\{[^}]+\})', section)
            overall_sentiment_distribution = {}
            if sentiment_match:
                try:
                    sentiment_str = sentiment_match.group(1).replace("'", '"')
                    overall_sentiment_distribution = ast.literal_eval(sentiment_str)
                except (SyntaxError, ValueError) as e:
                    print(f"Warning: Could not parse sentiment distribution: {e}")
            
            # Extract average sentiment
            avg_sentiment_match = re.search(r'Average Sentiment:\s*([\d.]+)', section)
            average_sentiment = float(avg_sentiment_match.group(1)) if avg_sentiment_match else 0.0
            
            # Extract categories with fixed pattern
            categories = {}
            category_blocks = re.split(r'\n(?=[\w\s-]+:)\n', section)
            
            for block in category_blocks:
                category_match = re.match(r'([\w\s-]+):', block)
                if not category_match or category_match.group(1).strip() == "Overall Period Summary":
                    continue
                    
                category_name = category_match.group(1).strip()
                
                # Extract messages count
                messages_match = re.search(r'Messages:\s*(\d+)', block)
                messages = int(messages_match.group(1)) if messages_match else 0
                
                # Extract sentiment distribution
                sentiment_match = re.search(r'Sentiment Distribution:\s*(\{[^}]+\})', block)
                sentiment_distribution = {}
                if sentiment_match:
                    try:
                        sentiment_str = sentiment_match.group(1).replace("'", '"')
                        sentiment_distribution = ast.literal_eval(sentiment_str)
                    except (SyntaxError, ValueError) as e:
                        print(f"Warning: Could not parse category sentiment distribution: {e}")
                
                # Extract summary with improved pattern
                summary_match = re.search(r'Summary:\s*(.*?)(?=\n\n|$)', block, re.DOTALL)
                summary = summary_match.group(1).strip() if summary_match else ""
                
                categories[category_name] = {
                    "messages": messages,
                    "sentimentDistribution": sentiment_distribution,
                    "summary": summary
                }
            
            # Extract overall summary with fixed pattern
            overall_summary_match = re.search(r'Overall Period Summary:[\s\n]*(.*?)(?=(?:\n===)|$)', section, re.DOTALL)
            overall_summary = ""
            if overall_summary_match:
                overall_summary = overall_summary_match.group(1).strip()
            
            result[period] = {
                "totalMessages": total_messages,
                "overallSentimentDistribution": overall_sentiment_distribution,
                "averageSentiment": average_sentiment,
                "categories": categories,
                "overallSummary": overall_summary
            }
            
        except Exception as e:
            print(f"Warning: Error processing section: {str(e)}")
            continue
    
    return result

def process_data():
    try:
        # Read the input file
        with open('mockAnalysisData.js', 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Extract the template literal content with a more robust pattern
        match = re.search(r'const\s+mockData\s*=\s*`([^`]+)`', content, re.DOTALL)
        if not match:
            raise ValueError('Could not find template literal in mockAnalysisData.js')
        
        raw_data = match.group(1)
        
        # Transform the data
        transformed_data = transform_analysis_data(raw_data)
        
        if not transformed_data:
            raise ValueError("No data was successfully transformed")
        
        # Save to JSON file
        with open('analysisOutput.json', 'w', encoding='utf-8') as file:
            json.dump(transformed_data, file, indent=2, ensure_ascii=False)
        
        print("Data has been processed and saved to analysisOutput.json")
        
    except FileNotFoundError:
        print("Error: Could not find mockAnalysisData.js")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")
        print("Please check that the input file format matches the expected format")

if __name__ == "__main__":
    process_data()