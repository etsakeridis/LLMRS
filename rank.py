import pandas as pd
from random import shuffle
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

class Rank():
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.messages: list[dict[str, str]] = []

    def rank(self, user_profile_prompt: str, recommendations: list) -> list[int]:
        candidate_pool = "Candidate Pool:\n"
        for i, rec in enumerate(recommendations):
            candidate_pool +=(
                f"  - index: {i}\n"
                f"    title: {rec.title}\n"
                f"    genres: {rec.genres}\n"
            )

        self.messages.append({
            "role": "user",
            "content": (
                "~~~\n"
                f"{user_profile_prompt}\n"
                "~~~\n"
                "Based on the user's profile above, order the items in the candidate pool from best to worst recommendation for the user.\n"
                "Respond ONLY with a comma separated list of indices surrounded by triple curly braces (i.e. {{{1, 2, 3 ...}}}).\n"
                f"{candidate_pool}"
            )
        })

        completion: ChatCompletion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.5,
            top_p=0.7,
            max_tokens=1024
        )

        message = completion.choices[0].message.content
        list_start = message.find("{{{") + len("{{{")
        list_end = message.rfind("}}}")
        sorted_indices = list(map(int, message[list_start:list_end].split(", ")))

        return sorted_indices

def _test() -> None:
    rank = Rank(
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

    # Dummy recommendations
    n_recs = 10
    recs_df: pd.DataFrame = user_history_df.tail(n_recs).reset_index(drop=["Index"])
    recommendations = []
    for rec in recs_df.itertuples(index=False):
        recommendations.append(rec)
    shuffle(recommendations)

    user_history_df = user_history_df[:-n_recs].tail(25).reset_index()

    history_prompt = "Movie History:\n"
    for movie in user_history_df.itertuples(index=False):
        history_prompt +=(
            f"  - title: {movie.title}\n"
            f"    genres: {movie.genres}\n"
            f"    my tags: {movie.tags}\n"
        )
    history_prompt += (
        "The user has watched and liked the above movies.\n"
        "They are given in the order that the user watched them with the one watched most recently being last on the list.\n"
    )

    print("Initial order:")
    for i, movie in enumerate(recommendations):
        print(f"\t{i + 1:2d}) {movie.title} [{movie.rating}]")

    print("LLM order:")
    for i, index in enumerate(rank.rank(history_prompt, recommendations)):
        print(f"\t{i + 1:2d}) {recommendations[index].title}")

    recommendations.sort(key=lambda movie: float(movie.rating), reverse=True)
    print("Actual order:")
    for i, movie in enumerate(recommendations):
        print(f"\t{i + 1:2d}) {movie.title} [{movie.rating}]")

    print("")
    print("Message given to the LLM")
    print(rank.messages[0]["content"])

if __name__ == "__main__":
    _test()
