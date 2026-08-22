import pandas as pd
from collections import Counter


# --------------------------------
# Load data
# --------------------------------

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


# --------------------------------
# Find user's preferred tags
# --------------------------------

def findTags(user_id):

    # Get user's borrowed books
    user_borrowed_books = borrow_history[
        borrow_history["user_id"] == user_id
    ]

    if user_borrowed_books.empty:
        print("User not found.")
        return []

    # Get book IDs
    book_ids = user_borrowed_books["book_id"].tolist()

    # Find those books in books.csv
    borrowed_books = books[
        books["book_id"].isin(book_ids)
    ]

    # Count the tags
    tag_count = Counter()

    for tags in borrowed_books["tags"]:
        for tag in tags.split(","):
            tag_count[tag.strip()] += 1

    # Convert tag counts to an array
    tag_array = []

    for tag, count in tag_count.most_common():
        tag_array.append([tag, count])

    return tag_array


# --------------------------------
# Calculate book recommendation score
# --------------------------------

def calculate_score(tags, tag_preferences):

    score = 0

    # Convert:[["adventure", 7], ["fantasy", 4]] into: ["fantasy", "adventure", "fiction"]

    tag_counts = dict(tag_preferences)

   

    for tag in tags.split(","):

        tag = tag.strip()

        if tag in tag_counts:
            score += tag_counts[tag]

    return score


# --------------------------------
# Generate recommendations
# --------------------------------

def recommend(user_id):

    # Find user's preferences
    tag_preferences = findTags(user_id)

    if not tag_preferences:
        return pd.DataFrame()

    # Make a copy so the original books DataFrame isn't modified
    recommendations = books.copy()

    # Calculate score for each book
    recommendations["score"] = recommendations["tags"].apply(
        lambda tags: calculate_score(tags, tag_preferences)
    )

    # Remove unavailable books
    recommendations = recommendations[
        recommendations["available"] > 0
    ]

    # Remove books with no matching tags
    recommendations = recommendations[
        recommendations["score"] > 0
    ]

    # Highest score first
    recommendations = recommendations.sort_values(
        by="score",
        ascending=False
    )

    return recommendations


# --------------------------------
# Main
# --------------------------------

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