from openai import OpenAI
import os
# load dotenv
from dotenv import load_dotenv
load_dotenv()
from prompts import gpt_35_simplemath

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def create_spurious_reasoning_1(
    problem
):
    message = gpt_35_simplemath.format(problem)
    completion = create_chat_completion_3(message)
    return completion.choices[0].message.content

def create_chat_completion_3(
    message,
    *_,
    **kwargs,
):
    """Create a chat completion using the OpenAI API with GPT 3.5

    Args:
        messages: A list of messages to feed to the chatbot.
        kwargs: Other arguments to pass to the OpenAI API chat completion call.
    Returns:
        OpenAIObject: The ChatCompletion response from OpenAI
    """

    model = "gpt-3.5-turbo"

    messages = [
        {
                "role": "system",
                "content": message,
        }
    ]
    completion = client.chat.completions.create(
        messages=messages,
        model=model,
        **kwargs,
    )

    if not hasattr(completion, "error"):
        print(f"Response: {completion}")

    return completion