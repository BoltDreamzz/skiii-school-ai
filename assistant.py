import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = os.getenv("AI_API_KEY", "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")
BASE_URL = os.getenv(
    "AI_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)
MODEL = os.getenv(
    "AI_MODEL",
    "openai/gpt-oss-20b",
)


if not API_KEY:
    raise RuntimeError(
        "AI_API_KEY is missing. Add it to your .env file."
    )


client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


# ============================================================
# ASSISTANT PERSONALITY
# ============================================================

SYSTEM_PROMPT = """
You are SKIII, a highly intelligent personal AI assistant.

Your job is to help the user with:
- programming
- debugging
- learning
- research
- planning
- technical explanations
- problem solving
- general questions

Behavior:
- Be accurate and practical.
- Explain difficult concepts clearly.
- When writing code, provide complete working examples.
- Do not unnecessarily repeat yourself.
- Remember relevant information from the conversation.
- If the user is confused, simplify the explanation.
- If there are multiple valid approaches, explain the important tradeoffs.
- Never pretend you performed an action that you did not perform.
"""


# ============================================================
# CONVERSATION MEMORY
# ============================================================

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
]


# ============================================================
# TERMINAL UI
# ============================================================

def banner():
    logo = r"""
        ███████╗██╗  ██╗██╗██╗██╗
        ██╔════╝██║ ██╔╝██║██║██║
        ███████╗█████╔╝ ██║██║██║
        ╚════██║██╔═██╗ ██║██║██║
        ███████║██║  ██╗██║██║██║
        ╚══════╝╚═╝  ╚═╝╚═╝╚═╝══╝
    """

    print()
    print(logo)

    print("    ┌──────────────────────────────────────┐")
    print(f"    │  Assistant  : SKIII AI               │")
    print(f"    │  Model      : {MODEL:<25}│")
    print(f"    │  Provider   : {BASE_URL:<25}│")
    print("    │  Memory     : Active                 │")
    print("    │  Streaming  : Enabled                │")
    print("    │  Status     : Online                 │")
    print("    └──────────────────────────────────────┘")
    print()
    print("    Type /help for commands.")
    print()


def show_help():
    print()
    print("Available commands:")
    print()
    print("  /help   - Show this help menu")
    print("  /clear  - Clear conversation memory")
    print("  /save   - Save conversation to a file")
    print("  /model  - Show current AI model")
    print("  /exit   - Exit the assistant")
    print()


# ============================================================
# SAVE CONVERSATION
# ============================================================

def save_conversation():
    os.makedirs("conversations", exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    filename = f"conversations/chat_{timestamp}.txt"

    with open(filename, "w", encoding="utf-8") as file:
        for message in messages:

            if message["role"] == "system":
                continue

            role = message["role"].upper()

            file.write(
                f"\n[{role}]\n"
                f"{message['content']}\n"
            )

    print(f"\nConversation saved to: {filename}\n")


# ============================================================
# ASK AI
# ============================================================

def ask_ai(user_input):

    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    try:

        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
            stream=True,
        )

        print("\nSKIII > ", end="", flush=True)

        full_response = ""

        for chunk in stream:

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                text = delta.content

                print(
                    text,
                    end="",
                    flush=True,
                )

                full_response += text

        print("\n")

        messages.append(
            {
                "role": "assistant",
                "content": full_response,
            }
        )

    except Exception as error:

        print()
        print("ERROR:")
        print(error)

        # Remove the unanswered user message
        # so the conversation remains clean.
        messages.pop()


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    banner()

    while True:

        try:

            user_input = input("You > ").strip()

        except KeyboardInterrupt:

            print("\n\nGoodbye.")
            break

        except EOFError:

            print("\n\nGoodbye.")
            break


        if not user_input:
            continue


        # ----------------------------------------------------
        # COMMANDS
        # ----------------------------------------------------

        if user_input == "/exit":
            print("\nGoodbye.")
            break


        if user_input == "/help":
            show_help()
            continue


        if user_input == "/clear":

            messages.clear()

            messages.append(
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            )

            print("\nConversation cleared.\n")
            continue


        if user_input == "/model":

            print()
            print(f"Current model: {MODEL}")
            print(f"Endpoint: {BASE_URL}")
            print()

            continue


        if user_input == "/save":

            save_conversation()
            continue


        # ----------------------------------------------------
        # NORMAL AI MESSAGE
        # ----------------------------------------------------

        ask_ai(user_input)


if __name__ == "__main__":
    main()
