from __future__ import annotations

import sys
import traceback
import discord

from discord.ext import commands

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from enfugue.discord.bot import EnfugueDiscordBot

class CommandErrorHandler(commands.Cog):
    """
    This handler is used for controlling error handling behavior.
    """
    def __init__(self, bot: EnfugueDiscordBot) -> None:
        self.bot = bot

    @commands.Cog.listener()
    async def on_command_error(
        self,
        ctx: commands.Context,
        error: commands.CommandError
    ) -> None:
        """
        What to do when an error occurs.
        """
        if isinstance(error, commands.CommandNotFound):
            await ctx.send('I do not know that command?!')
        else:
            print('Ignoring exception in command {}:'.format(ctx.command), file=sys.stderr)
            traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
