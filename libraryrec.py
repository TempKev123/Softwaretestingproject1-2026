import pandas as pd
from collections import Counter


# Load data
try:
    books = pd.read_csv("dummydata/books.csv")
except FileNotFoundError:
    print("Error: books.csv not found.")
    books = pd.DataFrame()

try:
    borrow_history = pd.read_csv("dummydata/userhistory.csv")
except FileNotFoundError:
    print("Error: userhistory.csv not found.")
    borrow_history = pd.DataFrame()


def findTags(user_id):
    user_borrowed_books = borrow_history[
        borrow_history["user_id"] == user_id
    ]

    if user_borrowed_books.empty:
        print(f"User {user_id} has no borrow history.")
        return [], set()

    borrowed_book_ids = set(user_borrowed_books["book_id"].tolist())

    borrowed_books = books[
        books["book_id"].isin(borrowed_book_ids)
    ]

    tag_count = Counter()
    for tags in borrowed_books["tags"].dropna():
        for tag in tags.split(","):
            tag_count[tag.strip()] += 1

    return tag_count.most_common(), borrowed_book_ids


def calculate_score(tags, tag_preferences):
    score = 0
    tag_counts = dict(tag_preferences)

    for tag in str(tags).split(","):
        tag = tag.strip()
        if tag in tag_counts:
            score += tag_counts[tag]

    return score


def recommend(user_id, top_n=5):
    # Find user's tag preferences and previously borrowed books
    tag_preferences, borrowed_book_ids = findTags(user_id)

    # Filter out unavailable books and books already read
    candidate_books = books[
        (books["available"] > 0) & 
        (~books["book_id"].isin(borrowed_book_ids))
    ].copy()

    # Default fallback: Popularity sort with score = 0
    if not tag_preferences or candidate_books.empty:
        candidate_books["score"] = 0
        return candidate_books.sort_values(by="popularity", ascending=False).head(top_n)

    # Calculate score for each book based on tag matches
    candidate_books["score"] = candidate_books["tags"].apply(
        lambda tags: calculate_score(tags, tag_preferences)
    )
    matching_books = candidate_books[candidate_books["score"] > 0]

    # If no books match the tags, fall back to overall top popular candidate books
    if matching_books.empty:
        return candidate_books.sort_values(by="popularity", ascending=False).head(top_n)

    # Sort by tag match score first, then popularity
    return matching_books.sort_values(
        by=["score", "popularity"],
        ascending=[False, False]
    ).head(top_n)


def test():
    users = ["U001", "U002", "U003", "U004", "U005"]

    for user_id in users:
        recommendations = recommend(user_id)

        print(f"\nRecommendations for {user_id}:")

        if recommendations.empty:
            print("No recommendations found.")
            continue

        print(
            recommendations[
                [
                    "book_id",
                    "title",
                    "author",
                    "score",
                    "available"
                ]
            ].head(3).to_string(index=False)
        )


if __name__ == "__main__":
    test()