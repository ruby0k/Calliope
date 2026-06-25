# Fixed prompts for behavioral eval of a GENERAL base LM (completion-style stems,
# not chat). Grouped by domain so per-category degeneration can be compared.
PROMPT_CATEGORIES = {
    "factual": [
        "The capital of France is",
        "The Earth orbits the",
        "Water is made up of hydrogen and",
        "The largest planet in the solar system is",
        "World War II ended in the year",
        "The chemical symbol for gold is",
        "Mount Everest is the tallest",
    ],
    "science": [
        "Photosynthesis is the process by which plants",
        "The theory of evolution explains how",
        "Gravity is the force that",
        "DNA carries the genetic information that",
        "The speed of light is approximately",
        "An atom is made up of protons, neutrons, and",
    ],
    "explanation": [
        "The main difference between a virus and a bacterium is",
        "The reason the sky appears blue is that",
        "In economics, supply and demand describes",
        "A computer's CPU is responsible for",
        "To solve a quadratic equation, you first",
        "The water cycle works by",
    ],
    "code": [
        "def fibonacci(n):",
        "import numpy as np\n\n",
        "# Function to check whether a number is prime\ndef is_prime(n):",
        "class Stack:\n    def __init__(self):",
        "def reverse_string(s):\n    return",
        "SELECT name, age FROM users WHERE",
    ],
    "continuation": [
        "In recent years, researchers have discovered that",
        "The history of the Roman Empire begins with",
        "According to the report, the main causes of climate change are",
        "One of the most important inventions of the twentieth century was",
        "The first step in learning a new programming language is",
    ],
}

PROMPTS = [prompt for prompts in PROMPT_CATEGORIES.values() for prompt in prompts]
