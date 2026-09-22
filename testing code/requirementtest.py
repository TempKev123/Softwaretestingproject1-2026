import libraryrec as lb

preferences, borrowed_books = lb.findTags("U001")
print("preferences:",preferences)

print("score =", lb.calculate_score("fantasy, Adventure",preferences))

print(lb.recommend("U001"))
print("_"*150)
for user_id in ["U002", "U003", "U004", "U005"]:
    recommendations = lb.recommend(user_id, 3)

    print(recommendations[["book_id", "title"]].to_string(index=False))

