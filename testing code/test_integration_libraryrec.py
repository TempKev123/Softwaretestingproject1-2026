"""
Integration Testing — libraryrec.py
=====================================
Unlike control flow / domain testing (which test one function at a time
with made-up inputs), integration testing checks that our 3 functions
work correctly TOGETHER, and that the whole pipeline works with the
real CSV files, not just isolated fake data.

Each test below matches one scenario in Integration_Testing_Report.docx
(Objective / Steps / Expected Result).

Run with:  python3 test_integration_libraryrec.py -v
"""

import unittest
import pandas as pd
import libraryrec as lr


def make_books(rows):
    return pd.DataFrame(rows, columns=["book_id", "title", "author", "tags", "available", "popularity"])


def make_history(rows):
    return pd.DataFrame(rows, columns=["user_id", "book_id", "borrow_date", "return_date"])


class TestScenario1_TagPreferenceFeedsIntoScoring(unittest.TestCase):
    """
    Scenario 1: findTags() -> calculate_score() integration

    Objective: verify that the tag preferences built by findTags() get
    correctly used by calculate_score() to score a DIFFERENT book.
    """

    def test_tags_from_findtags_score_correctly_on_other_book(self):
        lr.borrow_history = make_history([["U001", "B001", "2026-01-01", "2026-01-05"]])
        lr.books = make_books([
            ["B001", "Already Read", "Author A", "fantasy,adventure", 2, 90],
            ["B002", "Candidate Book", "Author B", "fantasy,adventure,epic", 3, 70],
        ])

        # Step 1+2: get real tag preferences from findTags()
        tag_preferences, borrowed_ids = lr.findTags("U001")
        self.assertEqual(sorted(t for t, c in tag_preferences), ["adventure", "fantasy"])

        # Step 3: feed those real preferences into calculate_score() for a different book
        score = lr.calculate_score("fantasy,adventure,epic", tag_preferences)

        # Step 4: score should reflect exactly the 2 overlapping tags
        self.assertEqual(score, 2)


class TestScenario2_BorrowedBooksExcludedFromRecommend(unittest.TestCase):
    """
    Scenario 2: findTags() -> recommend() integration

    Objective: verify that the borrowed_book_ids returned by findTags()
    actually get excluded when recommend() builds its candidate pool.
    """

    def test_borrowed_book_id_excluded_end_to_end(self):
        lr.borrow_history = make_history([["U001", "B001", "2026-01-01", "2026-01-05"]])
        lr.books = make_books([
            ["B001", "Already Read", "Author A", "fantasy,adventure", 2, 90],
            ["B002", "New Book", "Author B", "fantasy,adventure", 2, 70],
        ])

        # Step 1+2: confirm findTags() itself reports B001 as borrowed
        tag_preferences, borrowed_ids = lr.findTags("U001")
        self.assertIn("B001", borrowed_ids)

        # Step 3+4: confirm recommend() actually leaves B001 out of the final list
        result = lr.recommend("U001")
        self.assertNotIn("B001", result["book_id"].values)
        self.assertIn("B002", result["book_id"].values)


class TestScenario3_FullPipelineWithRealFiles(unittest.TestCase):
    """
    Scenario 3: CSV files -> findTags() -> calculate_score() -> recommend()

    Objective: verify the full pipeline works end-to-end using the real
    dummydata/books.csv and userhistory.csv files, not synthetic test data.
    """

    @classmethod
    def setUpClass(cls):
        cls.real_books = pd.read_csv("dummydata/books.csv")
        cls.real_history = pd.read_csv("dummydata/userhistory.csv")

    def setUp(self):
        lr.books = self.real_books.copy()
        lr.borrow_history = self.real_history.copy()

    def test_real_existing_user_gets_valid_recommendations(self):
        # Step 1: real files already loaded in setUp
        # Step 2: call recommend() for a real user that exists cleanly in the data
        result = lr.recommend("U001")

        # Step 3: verify basic correctness of the real end-to-end result
        self.assertFalse(result.empty)
        self.assertTrue((result["available"] > 0).all())

        # borrowed books for U001 in the real file must not appear in the result
        borrowed = set(self.real_history[self.real_history["user_id"] == "U001"]["book_id"])
        overlap = borrowed.intersection(set(result["book_id"]))
        self.assertEqual(overlap, set())

    def test_real_unknown_user_gets_popularity_fallback_not_a_crash(self):
        # Step 4: call recommend() for a user_id that doesn't exist anywhere
        result = lr.recommend("U_DOES_NOT_EXIST")

        # Step 5: verify the fallback kicks in instead of an empty result or a crash
        self.assertFalse(result.empty)


class TestScenario4_MultipleUsersDontInterfere(unittest.TestCase):
    """
    Scenario 4: recommend() called for different users back-to-back

    Objective: verify calling recommend() for one user doesn't leak
    state into another user's results (no shared/mutated data between calls).
    """

    def test_repeated_calls_stay_consistent_across_users(self):
        lr.borrow_history = make_history([
            ["U001", "B001", "2026-01-01", "2026-01-05"],
            ["U002", "B002", "2026-01-01", "2026-01-05"],
        ])
        lr.books = make_books([
            ["B001", "Book One", "Author A", "fantasy", 2, 90],
            ["B002", "Book Two", "Author B", "romance", 2, 80],
            ["B003", "Book Three", "Author C", "fantasy", 2, 60],
        ])

        # Step 1: call recommend() for U001 first
        result_u001_first = lr.recommend("U001")

        # Step 2: call recommend() for a different user (U002) in between
        result_u002 = lr.recommend("U002")

        # Step 3: call recommend() for U001 again
        result_u001_second = lr.recommend("U001")

        # Step 4: U001's result must be identical both times, unaffected by U002's call
        self.assertListEqual(
            result_u001_first["book_id"].tolist(),
            result_u001_second["book_id"].tolist()
        )
        # sanity check the two users didn't get each other's excluded books mixed up
        self.assertNotIn("B001", result_u001_first["book_id"].values)
        self.assertNotIn("B002", result_u002["book_id"].values)


class TestScenario5_RobustnessToMessyRealWorldData(unittest.TestCase):
    """
    Scenario 5: pipeline behavior when the data itself has quality issues

    Objective: verify the full pipeline still returns a sensible result,
    instead of crashing, when real-world data problems show up (a missing
    tags value, or a user_id that doesn't exactly match due to whitespace).
    """

    def test_missing_tags_value_does_not_crash_the_pipeline(self):
        lr.borrow_history = make_history([["U001", "B001", "2026-01-01", "2026-01-05"]])
        lr.books = make_books([
            ["B001", "No Tags Book", "Author A", None, 2, 90],
            ["B002", "Normal Book", "Author B", "fantasy", 2, 70],
        ])

        # Step 1+2: run the full pipeline where the borrowed book has no tags at all
        result = lr.recommend("U001")

        # Step 3: pipeline must not crash, and should still return a sensible fallback
        self.assertFalse(result.empty)

    def test_whitespace_polluted_user_id_falls_back_instead_of_crashing(self):
        # Using the real CSV file, which we know has whitespace-polluted user_ids
        real_books = pd.read_csv("dummydata/books.csv")
        real_history = pd.read_csv("dummydata/userhistory.csv")
        lr.books = real_books
        lr.borrow_history = real_history

        # Step 4: call recommend() using the CLEAN version of a user_id that
        # only exists in the messy/whitespace-polluted form in the real file
        result = lr.recommend("U002")

        # pipeline should not crash; it should just treat U002 as unknown and
        # fall back to popularity instead of erroring out
        self.assertFalse(result.empty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
