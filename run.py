import logging

from griptape.chunkers import MarkdownChunker
from griptape.drivers import BasePromptDriver, OpenAiChatPromptDriver
from griptape.structures import Workflow
from griptape.tasks import PromptTask, TextSummaryTask, ToolkitTask
from griptape.tokenizers import OpenAiTokenizer
from griptape.tools import FileManager, WebScraper


# models seem to do better when smart quotes are converted to ascii, especially
# when identifying quotes in the text.
def convert_smart_quotes_to_ascii(text):
    text = text.replace("“", '"').replace("”", '"')  # Replace double smart quotes
    text = text.replace("‘", "'").replace("’", "'")  # Replace single smart quotes
    return text


# TODO click w/cli arguments

target_file = "research.md"
with open(target_file, "r") as file:
    content_to_summarize = file.read()

target = "research_prompt.md"
with open(target, "r") as file:
    summarization_prompt = file.read()

content_to_summarize = convert_smart_quotes_to_ascii(content_to_summarize)

tokenizer = OpenAiTokenizer(model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_4_MODEL)

# the summarization prompt will be added to each chunk, so let's count it and leave some space for a response
response_token_buffer = 400
summarization_prompt_token_count = tokenizer.count_tokens(summarization_prompt)
max_tokens = (
    tokenizer.max_tokens - summarization_prompt_token_count - response_token_buffer
)

# split the input content into chunks that can be summarized
content_to_summarize_chunks = MarkdownChunker(
    tokenizer=tokenizer, max_tokens=max_tokens
).chunk(content_to_summarize)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(f"chunk size chunks={len(content_to_summarize_chunks)}")

workflow = Workflow(
    prompt_driver=OpenAiChatPromptDriver(
        model=tokenizer.model,
        temperature=0,
    )
)

meta_summarization_task = PromptTask(
    """
Combine the notes below:
{% for key, value in parent_outputs.items() %}

---
{{ value }}

{% endfor %}
  """
)

for content_chunk in content_to_summarize_chunks:
    text_chunk = content_chunk.value
    summarize_task = PromptTask(f"{summarization_prompt}\n{text_chunk}")

    workflow.add_task(summarize_task)

    # TODO need to conditionally allow for summarization if the type of prompt allows for it
    if len(content_to_summarize_chunks) > 1:
        summarize_task.add_child(meta_summarization_task)

result = workflow.run()

# TODO this seems wrong
print(result[0].output.value)
