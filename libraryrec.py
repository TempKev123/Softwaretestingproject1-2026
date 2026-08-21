def recommend_books(user_id):
    import pandas as pd
    from collections import Counter

    # Load in the data
    try:
        books = pd.read_csv("dummydata/books.csv")
    except FileNotFoundError:
        print("Error: books.csv not found.")
        return []

    try:
        borrow_history = pd.read_csv("dummydata/userhistory.csv")
    except FileNotFoundError:
        print("Error: userhistory.csv not found.")
        return []

    # Get user's borrowed books
    user_borrowed_books = borrow_history[
        borrow_history['user_id'] == user_id
    ]

    if user_borrowed_books.empty:
        print("User not found.")
        return []

    # Get book IDs
    book_ids = user_borrowed_books['book_id'].tolist()

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


def main():
    # Main driver allows code testing in file and importing to other files
    print(recommend_books("U001"))
    print(recommend_books("U002"))
    print(recommend_books("U003"))
    print(recommend_books("U004"))
    print(recommend_books("invalid_user"))


if __name__ == "__main__":
    main()