import logging

import click
from griptape.chunkers import MarkdownChunker
from griptape.drivers import OpenAiChatPromptDriver
from griptape.structures import Workflow
from griptape.tasks import PromptTask
from griptape.tokenizers import OpenAiTokenizer


def convert_smart_quotes_to_ascii(text):
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return text


def main(content_to_summarize_path, summarization_prompt_path):
    with open(content_to_summarize_path, "r") as file:
        content_to_summarize = file.read()

    with open(summarization_prompt_path, "r") as file:
        summarization_prompt = file.read()

    content_to_summarize = convert_smart_quotes_to_ascii(content_to_summarize)

    tokenizer = OpenAiTokenizer(model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_4_MODEL)

    response_token_buffer = 400
    summarization_prompt_token_count = tokenizer.count_tokens(summarization_prompt)
    max_tokens = (
        tokenizer.max_tokens - summarization_prompt_token_count - response_token_buffer
    )

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

        if len(content_to_summarize_chunks) > 1:
            summarize_task.add_child(meta_summarization_task)

    result = workflow.run()

    print(result.output_task.output.value)


@click.command()
@click.argument(
    "content",
    type=click.Path(exists=True),
    help="Path to file containing content to summarize.",
)
@click.argument(
    "prompt", type=click.Path(), help="Path to file containing summarization prompt"
)
def cli(content, prompt):
    main(content, prompt)


if __name__ == "__main__":
    cli()
