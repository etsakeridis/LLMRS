import yaml
from yaml.scanner import ScannerError
from openai import OpenAI
# Technically I shouldn't be using _streaming but I wanted the intellisense.
from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from colorama import Fore, Style, just_fix_windows_console
just_fix_windows_console()

MAX_CHAT_ROUNDS = 5
DEFAULT_SYSTEM_MESSAGE = f'''
You are a friendly AI conversing with a user who wants to find a new movie to watch.
Your job is to construct the user's profile and give it to a recommendation system.
I will give you the user's message and you must respond with a yaml object.
Always respond with a single yaml object that contains the fields:
  n: current message number
  content: the message you wish to send to the user in multiline yaml format
  done: |
    set to true if you are done with the conversation and believe
    that you have completed the user's profile. Leave false otherwise.
  profile: json object where you collect information about the user
    sex: male or female
    age: if the user doesn't give an exact age then use the lowest valid approximation
    movies: list of movies the user likes; only list movies
    genres: list of genres the user likes; available genres are:
      - Action
      - Adventure
      - Animation
      - Children's
      - Comedy
      - Crime
      - Documentary
      - Drama
      - Fantasy
      - Film-Noir
      - Horror
      - Musical
      - Mystery
      - Romance
      - Sci-Fi
      - Thriller
      - War
      - Western
    directors: list of directors the user likes
    actors: list of actors the user likes
    time period: denotes the acceptable release dates for the user; update it based on the movies the user gives
      start
      end
Make sure to inquire about all the fields in the profile from the user.
If the user doesn't give much information or is uncertain about something provide examples and hypotheticals.
You may not suggest any movies but you may ask the user's opinion on them.
You should stop once you are done making the user's profile, after {MAX_CHAT_ROUNDS} messages (n={MAX_CHAT_ROUNDS}), or if the user asks you to.
In that case, let the user know when you are done making their profile and ask them if there is anything else they would like to add or discuss.
If there is not set the "done" field to true and do any other changes you want to the yaml profile.
Make sure you keep your responses in yaml format since they will be parsed by a program.
'''.strip()

DEFAULT_FIRST_MESSAGE = f'''
response:
  n: 1/{MAX_CHAT_ROUNDS}
  content: |
    I'm ready to help you find a new movie to watch.
    Please tell me a little about yourself, such as your sex, age, movies/genres/actors/directors you may like, whether you prefer old or new movies.
  done: false
  profile:
    sex: unknown
    age: 1
    movies:
    genres:
    directors:
    actors:
    time period:
      start: 0
      end: 9999
'''.strip()

class Conversation():
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.user_prefix = f"{Style.RESET_ALL}{Fore.RED}<USER>\n{Fore.RED}{Style.BRIGHT}"
        self.assistant_prefix = f"{Style.RESET_ALL}{Fore.BLUE}<ASSISTANT>\n{Fore.BLUE}{Style.BRIGHT}"
        self.system_message = DEFAULT_SYSTEM_MESSAGE
        self.first_message = DEFAULT_FIRST_MESSAGE
        self.messages: list[dict[str, str]] = []

    def converse(self, multiline_input: bool = False) -> dict:
        try:
            print(Style.BRIGHT + Fore.MAGENTA)
            if multiline_input:
                print("[TIP: End your input with Ctrl-Z+Return on Windows and Ctrl-D on *nix at the start of a new line)]")
            print("[TIP: You can end the conversation at any time by sending \"!q\" as your message.]")
            print(Style.RESET_ALL)

            self.messages.append({"role": "system", "content": self.system_message})
            self.messages.append({"role": "assistant", "content": self.first_message})
            print(f"{self.assistant_prefix}{self.first_message}", end="", flush=True)

            # Interactive Loop
            while True:
                # User Input
                print("")
                if multiline_input:
                    input_lines = []
                    content = "[ERROR GETTING USER INPUT]"
                    try:
                        input_lines.append(input(self.user_prefix))
                        while True:
                            input_lines.append(input())
                    except EOFError:
                        content = '\n'.join(input_lines)
                else:
                    content = input(self.user_prefix)
                if content == "!q":
                    break
                self.messages.append({"role": "user", "content": f"{content}"})

                # Get LLM Response Stream
                completion: Stream[ChatCompletionChunk] = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=0.5,
                    top_p=0.7,
                    max_tokens=1024,
                    stream=True
                )

                # Print LLM Response
                tokens = []
                print(self.assistant_prefix, end="", flush=True)
                for chunk in completion: # pylint: disable=not-an-iterable
                    choice = chunk.choices[0]
                    if choice.finish_reason is not None:
                        continue
                    token = choice.delta.content
                    tokens.append(token)
                    print(token, end="", flush=True)

                self.messages.append({
                    "role": "assistant",
                    "content": ''.join(tokens)
                })
        finally:
            print(Style.RESET_ALL, end="")

        # Extract User Profile
        n = 1
        while True:
            try:
                message_yaml = self.messages[-n]["content"]
            except ScannerError:
                n += 1
                continue
            break
        profile = yaml.safe_load(message_yaml)["response"]["profile"]

        return profile

def _test() -> None:
    conversation = Conversation(
        client=OpenAI(
            api_key="API_KEY",
            base_url="http://127.0.0.1:8080/v1"
        ),
        model="Meta-Llama-3-8B-Instruct-8.0bpw-exl2"
    )

    user_profile = conversation.converse()
    print(user_profile)

if __name__ == "__main__":
    _test()
