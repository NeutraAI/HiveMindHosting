import discord
import asyncio
from discord.ext import commands
from hivemind import RemoteExpert

# Replace with your own bot token
bot_token = ''
server_ip = ''

bot = commands.Bot(command_prefix="!")

@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")

@bot.command()
async def generate(ctx, *, prompt):
    try:
        async with RemoteExpert('expert.1', [server_ip]) as expert:
            generated_text = await expert.generate_text(prompt)
            await ctx.send(generated_text)
    except Exception as e:
        await ctx.send(f"An error occurred: {e}")

bot.run(bot_token)
