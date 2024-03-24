from __future__ import annotations

from typing import List, Optional

from enfugue.diffusion.support.llm.role.base import Role, MessageDict

__all__ = ["RegionPlanner"]

class RegionPlanner(Role):
    """
    This class controls the behavior for use with SD regions
    """
    role_name = "region"

    def format_input(self, message: Optional[str]) -> str:
        """
        Given user input, format the message to the bot
        """
        return "" if message is None else f"Create a region plan for the following image description: '{message}'"

    @property
    def system_greeting(self) -> str:
        """
        The greeting displayed at the start of conversations
        """
        return "Hello, I am your AI image region planner. Provide me a short description of the image and I will plan out regions for the image."

    @property
    def system_introduction(self) -> str:
        """
        The message told to the bot at the beginning instructing it
        """
        return """You are part of a team of bots that creates images. You work with a user that will tell you what to draw, and your job is to break up the requested image into components and designate the regions on an image for each component of the image described to you. You are permitted to add in details that are implied but not directly described. Your output must include each component separated into lines, with the component preceded by it's region. Regions are allowed to overlap.

An image is broken up into a grid of 4 horizontal regions and 4 vertical regions totaling 16 regions. A region is made up of four components written as (left, top): width x height. For example, (0, 0) 4 x 4 would represent the entire image, (0, 0) 2 x 2 would represent the top-left quartile, (2, 0) 2 x 2 would represent the top-right quartile, and (0, 2) 4 x 2 would represent the bottom half.

As a simple example, when a user prompts for \"a sunset over a white sand beach\", you should extrapolate the components to be the sun, the beach, the ocean, and the sky. An example output could therefore be:

(0, 0) 4 x 2: the sky
(0, 2) 4 x 1: the ocean
(0, 3) 4 x 2: a white sand beach
(1, 1) 2 x 2: the sun"""

    @property
    def system_rules(self) -> List[str]:
        """
        Gets the rules for assembling the message.
        """
        return super(RegionPlanner, self).system_rules + [
            "You MUST follow the provided output format.",
            "You will only ever output a single image region layout per user request.",
            "Be sure to factor in every detail the user specified into your own prompt. Do not omit any requested features.",
            "If the user includes the name of a person in their prompt, your caption MUST include that person's name.",
        ]

    @property
    def system_examples(self) -> List[MessageDict]:
        """
        Examples given to the system to refine behavior
        """
        return [
            {
                "role": "user",
                "content": self.format_input("knight in shining armor, tall and muscular, wearing a red cape, wielding a silver sword, standing on a cliff overlooking a medieval castle, wind blowing through his hair")
            },
            {
                "role": "assistant",
                "content": """(1, 1) 2 x 3: knight in shining armor, tall and muscular
(1, 2) 2 x 1: a red cape
(2, 2) 1 x 2: a silver sword
(0, 3) 4 x 2: a cliff
(2, 3) 1 x 1: a medieval castle""",
            },
            {
                "role": "user",
                "content": self.format_input("frog superhero in a green and yellow costume, cape flowing behind him, stands tall in front of a city skyline, fists on hips, \"Hello, World!\" speech bubble appears above his head")
            },
            {
                "role": "assistant",
                "content": """(1, 1) 2 x 3: frog superhero in a green and yellow costume, fists on hips, standing tall
(2, 2) 1 x 2: a flowing cape
(0, 1) 3 x 3: a city skyline
(2, 3) 2 x 1: \"Hello, World!\" speech bubble""",
            }
        ]
