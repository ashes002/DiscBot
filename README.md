# Overview

Alpaca Roleplay Discordbot is a software project for running the Alpaca (or LLaMa) Large Language Model as a roleplaying discord bot. The bot is designed to run locally on a PC with as little as 8GB of VRAM. The bot listens for messages mentioning its username, replying to it's messages, or any DM's it receives, processes the message content, and generates a response based on the input.

REQUIRED: NVIDIA GPU with at least 12GB of VRAM for 7B model, and 24GB of VRAM for 13B models

I'm now recommending the use of Alpaca-Native Finetune on 7B model, requiring only 12GB vram: [https://huggingface.co/chavinlo/alpaca-native](https://huggingface.co/chavinlo/alpaca-native)

You can set the chat history to anything you like with !limit, but the LLAMA models can only handle 2,000 tokens of input for any given prompt, so be sure to set it low if you have a large character card.

This bot utilizes a json file some may know as a character card, to place into it's preprompt information about the character it is to role play as. You can manually edit the json or use a tool like [https://zoltanai.github.io/character-editor/](https://zoltanai.github.io/character-editor/) to make yourself a character card. For now, we only support one character at a time, and the active character card file should be character.json
