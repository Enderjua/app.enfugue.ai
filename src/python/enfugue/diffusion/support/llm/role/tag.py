from __future__ import annotations

from typing import List, Optional

from enfugue.diffusion.support.llm.role.base import Role

__all__ = ["DanbooruTagGenerator"]

class DanbooruTagGenerator(Role):
    """
    This class controls the behavior for use with SD
    """
    role_name = "tag"

    def format_input(
        self,
        message: Optional[str],
        rating: Optional[str]=None,
        artist: Optional[str]=None,
        characters: Optional[str]=None,
        copyrights: Optional[str]=None,
        aspect_ratio: Optional[float]=None,
        target: Optional[Literal["very_short", "short", "long", "very_long"]]=None,
        **kwargs: Any
    ) -> str:
        """
        Given user input, format the message to the bot
        """
        return "" if message is None else f"""rating: {rating or '<|empty|>'}
artist: {artist.strip() if artist else '<|empty|>'}
characters: {characters.strip() if characters else '<|empty|>'}
copyrights: {copyrights.strip() if copyrights else '<|empty|>'}
aspect ratio: {f"{aspect_ratio:.1f}" if aspect_ratio else '1.0'}
target: {'<|' + target + '|>' if target else '<|long|>'}
general: {message.strip().strip(",")}<|input_end|>"""

    @property
    def use_system(self) -> bool:
        """
        Whether to use the system prompts
        """
        return False

    @property
    def use_chat(self) -> bool:
        """
        Whether to use the conversation structure
        """
        return False
