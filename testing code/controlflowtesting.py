"""
Control Flow Testing — libraryrec.py
=====================================
Each test method below corresponds to one independent (basis) path
identified in CFG_Node_Mapping.md. Node/path numbers in the docstrings
match that file and the Word report.

Run with:  python3 test_libraryrec.py -v
"""

import unittest
import pandas as pd
import io
import contextlib
import libraryrec as lr


def make_books(rows):
    return pd.DataFrame(rows, columns=["book_id", "title", "author", "tags", "available"])


def make_history(rows):
    return pd.DataFrame(rows, columns=["user_id", "book_id", "borrow_date", "return_date"])


class TestFindTags(unittest.TestCase):
    """V(G) = 5 -> 5 independent paths"""

    def test_path1_user_not_found(self):
        """Path 1-2(T)-3: user_id has no borrow history at all."""
        lr.borrow_history = make_history([["U001", "B001", "2026-01-01", "2026-01-05"]])
        lr.books = make_books([["B001", "Hobbit", "Tolkien", "fantasy,adventure", 3]])

        with contextlib.redirect_stdout(io.StringIO()) as out:
            result = lr.findTags("U999")

        self.assertEqual(result, [])
        self.assertIn("User not found.", out.getvalue())

    def test_path2_history_but_no_matching_books(self):
        """Path 1-2(F)-4-5-6-7(F)-10-11(F)-13: user has borrow records,
        but none of the borrowed book_ids exist in books.csv (data
        integrity edge case -> borrowed_books ends up empty -> both
        loops run zero times)."""
        lr.borrow_history = make_history([["U001", "B999", "2026-01-01", "2026-01-05"]])
        lr.books = make_books([["B001", "Hobbit", "Tolkien", "fantasy,adventure", 3]])

        result = lr.findTags("U001")
        self.assertEqual(result, [])

    def test_path3_single_book_single_tag(self):
        """Path 1-2(F)-4-5-6-7(T)-8(T)-9-8(F)-7(F)-10-11(T)-12-11(F)-13:
        one borrowed book with exactly one tag."""
        lr.borrow_history = make_history([["U001", "B001", "2026-01-01", "2026-01-05"]])
        lr.books = make_books([["B001", "Solo Tag Book", "Author A", "adventure", 2]])

        result = lr.findTags("U001")
        self.assertEqual(result, [["adventure", 1]])

    def test_path4_single_book_multiple_tags(self):
        """Path 1-2(F)-4-5-6-7(T)-8(T)-9-8(T)-9-8(F)-7(F)-10-11(T)-12-11(T)-12-11(F)-13:
        one borrowed book, inner loop (node 8) repeats across multiple tags."""
        lr.borrow_history = make_history([["U001", "B001", "2026-01-01", "2026-01-05"]])
        lr.books = make_books([["B001", "Multi Tag Book", "Author A", "fantasy,adventure,fiction", 2]])

        result = lr.findTags("U001")
        result_tags = sorted(t for t, c in result)
        self.assertEqual(result_tags, ["adventure", "fantasy", "fiction"])
        self.assertTrue(all(c == 1 for t, c in result))

    def test_path5_multiple_books_same_tag_aggregates(self):
        """Path 1-2(F)-4-5-6-7(T)-8(T)-9-8(F)-7(T)-8(T)-9-8(F)-7(F)-10-11(T)-12-11(F)-13:
        outer loop (node 7) repeats across two books; shared tag count
        should aggregate to 2 rather than appearing as two entries."""
        lr.borrow_history = make_history([
            ["U001", "B001", "2026-01-01", "2026-01-05"],
            ["U001", "B002", "2026-02-01", "2026-02-05"],
        ])
        lr.books = make_books([
            ["B001", "Book One", "Author A", "adventure", 2],
            ["B002", "Book Two", "Author B", "adventure", 2],
        ])

        result = lr.findTags("U001")
        self.assertEqual(result, [["adventure", 2]])


class TestCalculateScore(unittest.TestCase):
    """V(G) = 3 -> 3 independent paths (path 1 is structurally
    infeasible for normal string input; see CFG_Node_Mapping.md)"""

    def test_path2_tag_not_in_preferences(self):
        """Path 1-2-3(T)-4-5(F)-3(F)-7: single tag, no match -> score 0."""
        score = lr.calculate_score("horror", [["fantasy", 5]])
        self.assertEqual(score, 0)

    def test_path3_tag_in_preferences(self):
        """Path 1-2-3(T)-4-5(T)-6-3(F)-7: single tag, matches -> score added."""
        score = lr.calculate_score("fantasy", [["fantasy", 5]])
        self.assertEqual(score, 5)

    def test_path4_mixed_tags_loop_repeats(self):
        """Path 1-2-3(T)-4-5(T)-6-3(T)-4-5(F)-3(F)-7: multiple tags,
        loop (node 3) repeats, one match one miss."""
        score = lr.calculate_score("fantasy,horror", [["fantasy", 5], ["adventure", 3]])
        self.assertEqual(score, 5)

    def test_infeasible_path1_note(self):
        """Path 1-2-3(F)-7 in the formal basis set: loop runs zero
        times. Documented as INFEASIBLE for string input, because
        "".split(",") returns [''] (one element), not an empty list.
        This test proves that even an empty tags string still enters
        the loop once (with tag == "")."""
        score = lr.calculate_score("", [["fantasy", 5]])
        self.assertEqual(score, 0)  # "" never matches a real tag_counts key


class TestRecommend(unittest.TestCase):
    """V(G) = 2 -> 2 independent paths"""

    def test_path1_no_preferences_returns_empty(self):
        """Path 1-2(T)-3: unknown/new user -> findTags returns [] ->
        recommend short-circuits to an empty DataFrame."""
        lr.borrow_history = make_history([["U001", "B001", "2026-01-01", "2026-01-05"]])
        lr.books = make_books([["B001", "Hobbit", "Tolkien", "fantasy,adventure", 3]])

        result = lr.recommend("U999")
        self.assertTrue(result.empty)

    def test_path2_full_pipeline(self):
        """Path 1-2(F)-4-5-6-7-8-9: user has preferences -> scoring,
        availability filter, score filter, and sort all execute."""
        lr.borrow_history = make_history([
            ["U001", "B001", "2026-01-01", "2026-01-05"],
            ["U001", "B005", "2026-02-01", "2026-02-05"],
        ])
        lr.books = make_books([
            ["B001", "Hobbit", "Tolkien", "fantasy,adventure", 3],
            ["B005", "Narnia", "Lewis", "magic", 2],
            ["B002", "Fellowship", "Tolkien", "fantasy,adventure,magic", 4],  # matches all 3 pref tags -> highest score
            ["B003", "Chamber", "Rowling", "fantasy,adventure,magic", 0],     # would score too, but unavailable
            ["B004", "1984", "Orwell", "dystopian,political", 5],            # available but score 0
        ])

        result = lr.recommend("U001")

        # unavailable book (B003) must be filtered out
        self.assertNotIn("B003", result["book_id"].values)
        # zero-score book (B004) must be filtered out
        self.assertNotIn("B004", result["book_id"].values)
        # B002 matches every preferred tag -> should score highest and sort first
        self.assertEqual(result.iloc[0]["book_id"], "B002")

    def test_finding_already_borrowed_books_are_recommended(self):
        """FINDING (not a basis path, but surfaced while testing path 2):
        recommend() never removes the books the user already borrowed
        from the candidate pool, so a book the user just returned can
        be recommended right back to them."""
        lr.borrow_history = make_history([["U001", "B001", "2026-01-01", "2026-01-05"]])
        lr.books = make_books([["B001", "Hobbit", "Tolkien", "fantasy,adventure", 3]])

        result = lr.recommend("U001")
        self.assertIn("B001", result["book_id"].values)  # documents current (unintended?) behavior


class TestRecommendIntegration(unittest.TestCase):
    """Integration check using the real dummydata/*.csv files, exercising
    both recommend() paths end-to-end against production data."""

    @classmethod
    def setUpClass(cls):
        cls.books = pd.read_csv("dummydata/books.csv")
        cls.history = pd.read_csv("dummydata/userhistory.csv")

    def setUp(self):
        lr.books = self.books.copy()
        lr.borrow_history = self.history.copy()

    def test_real_user_gets_recommendations(self):
        """U001 exists in userhistory.csv -> recommend() path 2."""
        result = lr.recommend("U001")
        self.assertFalse(result.empty)
        self.assertTrue((result["available"] > 0).all())
        self.assertTrue((result["score"] > 0).all())

    def test_unknown_user_gets_empty_dataframe(self):
        """U005 does not exist in userhistory.csv -> recommend() path 1."""
        result = lr.recommend("U005")
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
