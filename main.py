from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
import os
import logging
import time
import json
from typing import Dict, Optional
import asyncio
from datetime import datetime, timedelta
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class Query(BaseModel):
    question: str

# Simple in-memory cache
cache: Dict[str, Dict] = {}
# Rate limiting tracker
request_tracker: Dict[str, list] = {"timestamps": []}
# Maximum requests per minute
MAX_REQUESTS_PER_MINUTE = 10
# Cache expiration time (in seconds)
CACHE_EXPIRATION = 3600  # 1 hour

# Configure Gemini API if API key is available
api_key = os.getenv("GEMINI_API_KEY")
model = None

if api_key:
    try:
        genai.configure(api_key=api_key)
        
        # Set up the model with safety settings
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]
        
        model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        logger.info("Gemini API configured successfully")
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {str(e)}")
else:
    logger.warning("GEMINI_API_KEY not set. Running in fallback mode.")

# Fallback responses for common coding questions
FALLBACK_RESPONSES = {
    "binary search": """
# Binary Search Time Complexity

Binary search has a time complexity of O(log n), not O(n log n).

## Why Binary Search is O(log n):

Binary search works by repeatedly dividing the search interval in half:

1. Start with the entire array (n elements)
2. Compare the middle element with the target value
3. If the target matches the middle element, we're done
4. If the target is less than the middle element, search the left half
5. If the target is greater than the middle element, search the right half
6. Repeat until the target is found or the interval is empty

Each comparison eliminates half of the remaining elements:
- First comparison: n elements
- Second comparison: n/2 elements
- Third comparison: n/4 elements
- And so on...

The number of comparisons needed is log₂(n), which is why the time complexity is O(log n).

## Example:
For an array of 1,000,000 elements:
- A linear search (O(n)) might need up to 1,000,000 comparisons
- A binary search (O(log n)) needs at most log₂(1,000,000) ≈ 20 comparisons

This makes binary search extremely efficient for large sorted datasets.
""",
    "quicksort": """
# Quicksort Time Complexity

Quicksort has the following time complexities:

- **Best case**: O(n log n)
- **Average case**: O(n log n)
- **Worst case**: O(n²)

## How Quicksort Works:
1. Choose a pivot element from the array
2. Partition the array around the pivot (elements less than pivot go to the left, greater go to the right)
3. Recursively apply the above steps to the sub-arrays

## Why O(n log n) in average case:
- Each partition takes O(n) time
- The recursion creates a tree of height log n (on average)
- Total work: O(n) × O(log n) = O(n log n)

## Why O(n²) in worst case:
- If the pivot is always the smallest or largest element (e.g., in an already sorted array)
- The partition creates one empty subarray and one with n-1 elements
- This leads to a recursion depth of n instead of log n
- Total work: O(n) × O(n) = O(n²)

To avoid the worst case, strategies like choosing a random pivot or using "median-of-three" pivot selection are commonly used.
""",
    "merge sort": """
# Merge Sort Time Complexity

Merge sort has a consistent time complexity of O(n log n) for all cases (best, average, and worst).

## How Merge Sort Works:
1. Divide the array into two halves
2. Recursively sort each half
3. Merge the sorted halves back together

## Why O(n log n):
- The divide step takes O(1) time
- The merge step takes O(n) time to combine n elements
- The recursion creates a tree of height log n
- Total work: O(n) × O(log n) = O(n log n)

## Advantages:
- Stable sort (preserves the relative order of equal elements)
- Guaranteed O(n log n) performance regardless of input data
- Works well for linked lists

## Disadvantages:
- Requires O(n) extra space for the merging process
- May be slower than quicksort for arrays due to the overhead of creating temporary arrays

Merge sort is often preferred for external sorting when data doesn't fit in memory, and for stable sorting requirements.
""",
    "big o notation": """
# Big O Notation Explained

Big O notation is used to describe the performance or complexity of an algorithm, specifically the worst-case scenario.

## Common Big O Complexities (from fastest to slowest):

- **O(1)** - Constant Time: The operation takes the same amount of time regardless of the input size.
  *Example*: Accessing an array element by index.

- **O(log n)** - Logarithmic Time: The operation's time increases logarithmically as input size grows.
  *Example*: Binary search in a sorted array.

- **O(n)** - Linear Time: The operation's time increases linearly with input size.
  *Example*: Searching through an unsorted array.

- **O(n log n)** - Linearithmic Time: Common in efficient sorting algorithms.
  *Example*: Merge sort, Quicksort (average case).

- **O(n²)** - Quadratic Time: Often seen in algorithms with nested loops.
  *Example*: Bubble sort, Selection sort.

- **O(2^n)** - Exponential Time: The operation's time doubles with each addition to the input.
  *Example*: Recursive calculation of Fibonacci numbers.

- **O(n!)** - Factorial Time: Extremely slow algorithms.
  *Example*: Solving the traveling salesman problem with brute force.

## Rules for Big O:
1. Drop constants: O(2n) becomes O(n)
2. Drop lower-order terms: O(n² + n) becomes O(n²)
3. Consider the worst-case scenario

Big O helps developers compare algorithms and make informed decisions about which to use based on expected input sizes and performance requirements.
"""
}

def get_cache_key(question: str) -> str:
    """Generate a cache key from the question."""
    return hashlib.md5(question.lower().strip().encode()).hexdigest()

def check_rate_limit() -> bool:
    """Check if we're within rate limits."""
    current_time = time.time()
    # Remove timestamps older than 1 minute
    request_tracker["timestamps"] = [t for t in request_tracker["timestamps"] 
                                    if t > current_time - 60]
    
    # Check if we're under the limit
    return len(request_tracker["timestamps"]) < MAX_REQUESTS_PER_MINUTE

def update_rate_limit():
    """Update the rate limit tracker with current request."""
    request_tracker["timestamps"].append(time.time())

def get_fallback_response(question: str) -> Optional[str]:
    """Try to find a fallback response for common questions."""
    question_lower = question.lower()
    
    for key, response in FALLBACK_RESPONSES.items():
        if key in question_lower:
            return response
    
    return None

@app.get("/")
def home():
    return {"message": "Faang AI is running!"}

@app.post("/ask")
async def ask_ai(query: Query):
    try:
        question = query.question.strip()
        logger.info(f"Received question: {question}")
        
        if not question:
            return JSONResponse(
                status_code=400,
                content={"error": "Question cannot be empty"}
            )
        
        # Check cache first
        cache_key = get_cache_key(question)
        if cache_key in cache:
            cache_entry = cache[cache_key]
            # Check if cache entry is still valid
            if cache_entry["expires_at"] > time.time():
                logger.info(f"Cache hit for question: {question}")
                return cache_entry["response"]
        
        # Try to find a fallback response for common questions
        fallback = get_fallback_response(question)
        
        # If Gemini API is not configured or we're rate limited, use fallback
        if not model or not check_rate_limit():
            if fallback:
                logger.info(f"Using fallback response for: {question}")
                response_data = {"answer": fallback}
                
                # Cache the response
                cache[cache_key] = {
                    "response": response_data,
                    "expires_at": time.time() + CACHE_EXPIRATION
                }
                
                return response_data
            else:
                # If no fallback is available
                if not model:
                    return JSONResponse(
                        status_code=503,
                        content={"error": "AI service is not configured. Please try again later."}
                    )
                else:
                    return JSONResponse(
                        status_code=429,
                        content={"error": "Too many requests. Please try again in a minute."}
                    )
        
        # Update rate limit counter
        update_rate_limit()
        
        # Add a system prompt to focus on coding questions
        prompt = f"""You are an AI assistant specialized in answering coding and algorithm questions.
        Please provide a detailed, accurate, and educational response to the following question:
        
        {question}
        
        If the question is about time complexity, make sure to explain the reasoning clearly.
        Include examples where appropriate.
        """
        
        # Try to get response from Gemini API
        try:
            response = model.generate_content(prompt)
            
            # Check if response has text
            if not hasattr(response, 'text') or not response.text:
                logger.error("Empty response received from Gemini API")
                
                # Try fallback if available
                if fallback:
                    response_data = {"answer": fallback}
                else:
                    return JSONResponse(
                        status_code=500,
                        content={"error": "AI returned an empty response. Please try a different question."}
                    )
            else:
                response_data = {"answer": response.text}
            
            # Cache the response
            cache[cache_key] = {
                "response": response_data,
                "expires_at": time.time() + CACHE_EXPIRATION
            }
            
            return response_data
            
        except Exception as api_error:
            logger.error(f"Gemini API error: {str(api_error)}")
            
            # Try fallback if available
            if fallback:
                response_data = {"answer": fallback}
                
                # Cache the response
                cache[cache_key] = {
                    "response": response_data,
                    "expires_at": time.time() + CACHE_EXPIRATION
                }
                
                return response_data
            else:
                return JSONResponse(
                    status_code=500,
                    content={"error": f"AI service error: {str(api_error)}"}
                )
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process request: {str(e)}"}
        )

# Endpoint to check API status and rate limits
@app.get("/status")
def get_status():
    current_time = time.time()
    # Remove timestamps older than 1 minute
    request_tracker["timestamps"] = [t for t in request_tracker["timestamps"] 
                                    if t > current_time - 60]
    
    requests_in_last_minute = len(request_tracker["timestamps"])
    requests_remaining = MAX_REQUESTS_PER_MINUTE - requests_in_last_minute
    
    return {
        "status": "online",
        "api_configured": model is not None,
        "rate_limit": {
            "max_requests_per_minute": MAX_REQUESTS_PER_MINUTE,
            "requests_in_last_minute": requests_in_last_minute,
            "requests_remaining": requests_remaining
        },
        "cache_size": len(cache),
        "fallback_topics_available": list(FALLBACK_RESPONSES.keys())
    }

# Endpoint to clear cache (admin only - you might want to add authentication)
@app.post("/clear-cache")
def clear_cache():
    global cache
    cache_size = len(cache)
    cache = {}
    return {"message": f"Cache cleared. {cache_size} entries removed."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
