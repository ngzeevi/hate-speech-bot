import discord

from bot_token import TOKEN
from train import predict_message 


intents = discord.Intents.all()
client = discord.Client(intents=intents)


def is_hate(message):
    predictions = predict_message(message)
    return True if not all(predictions) else False


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    if is_hate(message.content):
        await message.add_reaction('\U0000274C')
        await message.reply('Please refrain from posting offensive language.')        


client.run(TOKEN)
