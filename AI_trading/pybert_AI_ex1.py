# A simple text classification example
def classify_sentiment(text):
    """
    A very basic sentiment classification function
    Demonstrates the concept of text classification
    """
    # Simple keywords for sentiment detection
    positive_words = ['love', 'amazing', 'great', 'excellent', 'wonderful']
    negative_words = ['terrible', 'boring', 'bad', 'awful', 'horrible']
    
    # Convert text to lowercase for easier matching
    text_lower = text.lower()
    
    # Count positive and negative words
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    # Determine sentiment
    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    else:
        return "Neutral"

# Test the sentiment classifier
test_texts = [
    "I love this movie, it's amazing!",
    "This film is terrible and boring.",
    "The movie was okay, nothing special."
]

# Classify and print results
for text in test_texts:
    sentiment = classify_sentiment(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}\n")
