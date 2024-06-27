import pandas as pd
from collections import namedtuple
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from colorama import Fore, Style, just_fix_windows_console
just_fix_windows_console()

class Explanation():
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.messages: list[dict[str, str]] = []

    def explain(self, history: pd.DataFrame, recommendation: namedtuple):
        history_prompt = "Movie History:\n"
        for movie in history.itertuples(index=False):
            history_prompt +=(
                f"  - title: {movie.title}\n"
                f"    genres: {movie.genres}\n"
                f"    my tags: {movie.tags}\n"
            )

        first_user_prompt = (
            f"{history_prompt}"
            "I have watched and liked the above movies.\n"
            "They are given in the order I watched them with the one I watch most recently being last on the list.\n"
            "Please suggest a movie for me to watch next."
        )
        assistant_rec = (
            "A fun challenge!\n"
            "After analyzing your movie preferences, I've identified some common themes and genres that you seem to enjoy.\n"
            "Considering these patterns, here's a movie suggestion for you:\n"
            f"  title: {recommendation.title}\n"
            f"  genres: {recommendation.genres}\n"
            "Give it a try and let me know what you think!"
        )

        self.messages.append({"role": "user", "content": first_user_prompt})
        self.messages.append({"role": "assistant", "content": assistant_rec})
        self.messages.append({"role": "user", "content": "Could you please explain why you think I should watch this movie?"})

        completion: ChatCompletion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.5,
            top_p=0.7,
            max_tokens=1024
        )

        return completion.choices[0].message.content

def _test() -> None:
    explanation = Explanation(
        client=OpenAI(
            api_key="API_KEY",
            base_url="http://127.0.0.1:8080/v1"
        ),
        model="Meta-Llama-3-8B-Instruct-8.0bpw-exl2"
    )

    USER_ID = 62
    movies_df = pd.read_csv("ml-latest-small/movies.csv")
    ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
    tags_df = pd.read_csv("ml-latest-small/tags.csv")
    # select based on user and rating
    user_liked_movies_df = (
        ratings_df
        .loc[(ratings_df["userId"] == USER_ID) & (ratings_df["rating"] > 3.5)]
    )
    # timestamp + title, genres
    user_movies_df = pd.merge(
        user_liked_movies_df,
        movies_df,
        on='movieId',
    ).reset_index(drop=True)
    # tags
    user_history_df = (
        pd.merge(
            user_movies_df,
            tags_df.drop(columns="timestamp"),
            how="left",
            on=["movieId", "userId"]
        )
        # every tag in a different row -> tuple of tags in a single row
        .drop(columns=["userId"])
        .map(str) # turn nans to strings
        .groupby(["movieId", "rating", "timestamp", "genres", "title"])
        .aggregate(set)
    )
    # tuple of tags -> comma separated string
    user_history_df["tag"] = user_history_df["tag"].map(', '.join).replace("nan", "-")
    # flatten columns
    user_history_df.columns = set(v for v in user_history_df.columns.values)
    user_history_df.reset_index(inplace=True)
    # final touches
    user_history_df["genres"] = user_history_df["genres"].map(
        lambda genres: str.replace(genres, '|', ", ")
    )
    user_history_df = (user_history_df
        .rename(columns={"tag": "tags"})
        .sort_values("timestamp", ascending=True)
    )

    # Dummy recommendation
    Row = namedtuple("Row", user_history_df.columns)
    rec_df = user_history_df.tail(1).reset_index(drop=["Index"]).iloc[0]
    rec = Row(*rec_df)

    user_history_df = user_history_df[:-1].tail(25).reset_index()

    llm_response = explanation.explain(user_history_df, rec)

    try:
        print(f"{Style.RESET_ALL}{Fore.RED}<USER>\n{Fore.RED}{Style.BRIGHT}")
        print(explanation.messages[0]["content"])
        print(f"{Style.RESET_ALL}{Fore.BLUE}<ASSISTANT>\n{Fore.BLUE}{Style.BRIGHT}")
        print(explanation.messages[1]["content"])
        print(f"{Style.RESET_ALL}{Fore.RED}<USER>\n{Fore.RED}{Style.BRIGHT}")
        print(explanation.messages[2]["content"])
        print(f"{Style.RESET_ALL}{Fore.BLUE}<ASSISTANT>\n{Fore.BLUE}{Style.BRIGHT}")
        print(llm_response)
    finally:
        print(Style.RESET_ALL)

if __name__ == "__main__":
    _test()
