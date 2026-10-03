"""
PROMPT TEMPLATES

Kept in a module with no third-party imports, for two reasons:

* The one guarantee this work has to make about the answer prompt is
  that adding sentiment-aware tone changed nothing else. That is a
  claim about a string, and it should be checkable without standing up
  FAISS, an embedding model and a Google API key to look at it.

* ``langchain_helper`` imports the whole stack at module level, so
  anything defined there is untestable in an environment where those
  packages are absent.
"""

# The prompt this project has always sent. Do not reformat: the blank
# lines are part of it, and the tone block below is inserted in a way
# that leaves this string untouched when there is no tone.
BASE_PROMPT_TEMPLATE = """
You are an AI Customer Support Assistant.

Answer ONLY using the provided context.

If the answer is not available in the context, reply exactly:

"I don't know based on the knowledge base."

Context:
{context}

Question:
{question}
"""

# The anchor the tone block is inserted after. Matching on the opening
# sentence rather than on a line number keeps this correct if the
# prompt is ever edited above it.
TONE_ANCHOR = "You are an AI Customer Support Assistant.\n"


def build_prompt_template(tone_instruction: str = "") -> str:
    """
    The answer prompt, with an optional style block.

    With no tone instruction this returns ``BASE_PROMPT_TEMPLATE``
    unchanged -- not an equivalent string, the same one -- so an
    ordinary question is answered by exactly the prompt that was sent
    before sentiment analysis existed.

    The tone block carries style only. It is inserted above the
    grounding rules so that "Answer ONLY using the provided context."
    and the refusal sentence still read as the operative instruction.
    """

    if not tone_instruction:

        return BASE_PROMPT_TEMPLATE

    return BASE_PROMPT_TEMPLATE.replace(
        TONE_ANCHOR,
        TONE_ANCHOR + "\n" + tone_instruction.strip() + "\n",
        1,
    )
