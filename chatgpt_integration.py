import cohere

# Set your Cohere API key
COHERE_API_KEY = "DHThvqaz2rHqkCWCni9ClDDvOhcQmYGQv8lhW7Te"

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

def analyze_responses(responses):
    """Analyze all responses and provide a mental health summary."""
    combined_responses = " ".join(responses)
    analysis_prompt = f"""
    Analyze the following responses from a mental health assessment:
    Responses: {combined_responses}
    
    Based on these responses, assess the user's mental state and provide a brief summary.
    Indicate whether the user may need to seek professional help or if the responses indicate general well-being.
    """
    
    response = co.chat(
        message=analysis_prompt,
        model="command-r",  # or "command-r-plus" if you have access
        temperature=0.7
    )
    
    return response.text.strip()