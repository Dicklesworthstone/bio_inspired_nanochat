"""
SmolTalk by HuggingFace. Good "general" conversational dataset.
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
We use the "smol" version, which is more appropriate for smaller models.
"""

from typing import Any, Dict, List, cast

from datasets import Dataset, load_dataset
from tasks.common import Task


class SmolTalk(Task):
    """ smol-smoltalk dataset. train is 460K rows, test is 24K rows. """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        if split not in ["train", "test"]:
            raise ValueError("SmolTalk split must be train|test")
        dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split=split, revision="03c2461").shuffle(seed=42)
        self.ds: Dataset = dataset
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = cast(Dict[str, Any], self.ds[index])
        messages = cast(List[Dict[str, Any]], row["messages"])
        # ---------------------------------------------------------------------
        # sanity checking asserts here
        # TODO: we could remove these asserts later, for now just don't want any footguns
        # there is an optional system message at the beginning
        if len(messages) < 1:
            raise ValueError("SmolTalk messages must have at least 1 message")
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:] # optional system message is OK
        else:
            rest_messages = messages
        if len(rest_messages) < 2:
            raise ValueError("SmolTalk messages must have at least 2 messages")
        for i, message in enumerate(rest_messages):
            # user and assistant alternate as user,assistant,user,assistant,...
            expected_role = "user" if i % 2 == 0 else "assistant"
            if message["role"] != expected_role:
                raise ValueError(f"Message {i} has role {message['role']} but should be {expected_role}")
            if not isinstance(message["content"], str):
                raise ValueError("Content must be a string")
        # ---------------------------------------------------------------------
        # create and return the Conversation object (ok to emit the system message too)
        conversation = {
            "messages": messages,
        }
        return conversation
