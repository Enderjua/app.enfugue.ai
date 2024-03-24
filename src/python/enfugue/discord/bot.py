from __future__ import annotations

from discord import Intents
from discord.ext.commands import Bot
from enfugue.util import logger

class EnfugueDiscordBot(Bot):
    async def on_ready(self) -> None:
        logger.info(getattr(self, "channel", None))
        logger.info(getattr(self, "channels", None))
        for guild in self.guilds:
            logger.info(f"{self.user.name} has connected to {guild.name}.")
            logger.info(getattr(guild, "channel", None))
            logger.info(getattr(guild, "channels", None))

    async def on_message(self, message) -> None:
        if self.user == message.author:
            return
        logger.info(f"{message.author}: {message.content} {message.attachments}")
        await message.channel.send("sup")

    @staticmethod
    def execute(
        token: str,
        command_prefix: str="$"
    ) -> None:
        """
        Builds a default bot and runs using a token.
        """
        import enfugue.discord.cogs as cogs
        import asyncio
        intents = Intents.default()
        intents.message_content = True
        bot = EnfugueDiscordBot(
            command_prefix=command_prefix,
            intents=intents
        )
        loop = asyncio.get_event_loop()
        for cog_name in cogs.__all__:
            loop.run_until_complete(bot.add_cog(getattr(cogs, cog_name)(bot)))
        bot.run(token)
