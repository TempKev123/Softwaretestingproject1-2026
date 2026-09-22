"""
Domain Testing — libraryrec.py
================================
Each test method corresponds to one boundary point identified in
Domain_Testing_Mapping.md (ON / OFF / typical points for each condition
in the code). This is a different technique from control flow testing —
here we're checking exact boundary values, not branch coverage.

Run with:  python3 test_domain_libraryrec.py -v
"""

import unittest
import pandas as pd
import libraryrec as lr


def make_books(rows):
    return pd.DataFrame(rows, columns=["book_id", "title", "author", "tags", "available", "popularity"])


def make_history(rows):
    return pd.DataFrame(rows, columns=["user_id", "book_id", "borrow_date", "return_date"])


class TestDomainAvailable(unittest.TestCase):
    """Domain 1: available > 0"""

    def test_off_point_available_zero_excluded(self):
        """OFF point: available = 0 -> book must be excluded from candidates."""
        lr.borrow_history = make_history([])
        lr.books = make_books([["B001", "Zero Stock", "Author A", "fiction", 0, 50]])

        result = lr.recommend("U999")
        self.assertNotIn("B001", result["book_id"].values)

    def test_on_point_available_one_included(self):
        """ON point: available = 1 (smallest in-domain value) -> book must be included."""
        lr.borrow_history = make_history([])
        lr.books = make_books([["B001", "One Copy", "Author A", "fiction", 1, 50]])

        result = lr.recommend("U999")
        self.assertIn("B001", result["book_id"].values)

    def test_typical_point_available_five_included(self):
        """Typical point: available = 5 -> book included, well inside the domain."""
        lr.borrow_history = make_history([])
        lr.books = make_books([["B001", "Plenty of Copies", "Author A", "fiction", 5, 50]])

        result = lr.recommend("U999")
        self.assertIn("B001", result["book_id"].values)


class TestDomainScore(unittest.TestCase):
    """Domain 2: score > 0"""

    def test_off_point_score_zero_not_a_match(self):
        """OFF point: score = 0 -> book should NOT show up in the ranked results,
        it should be treated as a non-match (falls to popularity list instead)."""
        score = lr.calculate_score("horror", [("fantasy", 5)])
        self.assertEqual(score, 0)

    def test_on_point_score_one_counts_as_match(self):
        """ON point: score = 1 (smallest possible match) -> counts as a real match."""
        score = lr.calculate_score("rare_tag", [("rare_tag", 1)])
        self.assertEqual(score, 1)
        self.assertGreater(score, 0)

    def test_typical_point_score_five(self):
        """Typical point: score = 5 -> clearly a match."""
        score = lr.calculate_score("fantasy", [("fantasy", 5)])
        self.assertEqual(score, 5)


class TestDomainBorrowHistorySize(unittest.TestCase):
    """Domain 3: number of borrow history rows (drives .empty check in findTags)"""

    def test_off_point_zero_rows_is_empty(self):
        """OFF point: 0 borrow rows -> treated as no history."""
        lr.borrow_history = make_history([])
        lr.books = make_books([["B001", "Book", "Author", "fiction", 2, 50]])

        tag_preferences, borrowed_ids = lr.findTags("U001")
        self.assertEqual(tag_preferences, [])

    def test_on_point_one_row_is_not_empty(self):
        """ON point: exactly 1 borrow row -> history exists, gets processed."""
        lr.borrow_history = make_history([["U001", "B001", "2026-01-01", "2026-01-05"]])
        lr.books = make_books([["B001", "Book", "Author", "fiction", 2, 50]])

        tag_preferences, borrowed_ids = lr.findTags("U001")
        self.assertEqual(tag_preferences, [("fiction", 1)])

    def test_typical_point_many_rows(self):
        """Typical point: several borrow rows -> normal case."""
        lr.borrow_history = make_history([
            ["U001", "B001", "2026-01-01", "2026-01-05"],
            ["U001", "B002", "2026-02-01", "2026-02-05"],
            ["U001", "B003", "2026-03-01", "2026-03-05"],
        ])
        lr.books = make_books([
            ["B001", "Book1", "Author", "fiction", 2, 50],
            ["B002", "Book2", "Author", "fiction", 2, 50],
            ["B003", "Book3", "Author", "fiction", 2, 50],
        ])
        tag_preferences, borrowed_ids = lr.findTags("U001")
        self.assertEqual(tag_preferences, [("fiction", 3)])


class TestDomainCandidatePoolSize(unittest.TestCase):
    """Domain 4: candidate pool size after available+not-borrowed filters"""

    def test_off_point_zero_candidates_falls_back(self):
        """OFF point: 0 candidates left (everything unavailable) -> must still
        return something sensible (popularity fallback), not crash."""
        lr.borrow_history = make_history([])
        lr.books = make_books([["B001", "Out of Stock", "Author", "fiction", 0, 50]])

        result = lr.recommend("U999")
        self.assertTrue(result.empty)  # nothing at all to recommend, correctly empty

    def test_on_point_one_candidate(self):
        """ON point: exactly 1 candidate available -> should be returned."""
        lr.borrow_history = make_history([])
        lr.books = make_books([["B001", "Only Book", "Author", "fiction", 1, 50]])

        result = lr.recommend("U999")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["book_id"], "B001")

    def test_typical_point_many_candidates(self):
        """Typical point: several candidates -> normal case."""
        lr.borrow_history = make_history([])
        lr.books = make_books([
            ["B001", "Book1", "Author", "fiction", 2, 60],
            ["B002", "Book2", "Author", "fiction", 2, 70],
            ["B003", "Book3", "Author", "fiction", 2, 80],
        ])
        result = lr.recommend("U999")
        self.assertEqual(len(result), 3)


class TestDomainMatchingBooksSize(unittest.TestCase):
    """Domain 5: matching books size (score > 0 subset)"""

    def test_off_point_zero_matches_falls_back_to_popularity(self):
        """OFF point: 0 books match the user's tags -> falls back to popularity."""
        lr.borrow_history = make_history([["U001", "B001", "2026-01-01", "2026-01-05"]])
        lr.books = make_books([
            ["B001", "Already Read", "Author", "fantasy", 2, 90],   # borrowed, excluded from pool
            ["B002", "No Match", "Author", "horror", 2, 70],
        ])
        result = lr.recommend("U001")
        self.assertFalse(result.empty)
        self.assertEqual(result.iloc[0]["book_id"], "B002")  # popularity fallback still returns it

    def test_on_point_one_match(self):
        """ON point: exactly 1 book matches -> that book gets returned."""
        lr.borrow_history = make_history([["U001", "B001", "2026-01-01", "2026-01-05"]])
        lr.books = make_books([
            ["B001", "Already Read", "Author", "fantasy", 2, 90],
            ["B002", "The Match", "Author", "fantasy", 2, 70],
            ["B003", "No Match", "Author", "horror", 2, 80],
        ])
        result = lr.recommend("U001")
        self.assertEqual(result.iloc[0]["book_id"], "B002")

    def test_typical_point_many_matches(self):
        """Typical point: multiple matching books -> ranked normally."""
        lr.borrow_history = make_history([["U001", "B001", "2026-01-01", "2026-01-05"]])
        lr.books = make_books([
            ["B001", "Already Read", "Author", "fantasy", 2, 90],
            ["B002", "Match A", "Author", "fantasy", 2, 70],
            ["B003", "Match B", "Author", "fantasy", 2, 85],
        ])
        result = lr.recommend("U001")
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["book_id"], "B003")  # higher popularity wins tie


class TestDomainTopN(unittest.TestCase):
    """Domain 6: top_n vs. number of available candidates (N)"""

    def setUp(self):
        lr.borrow_history = make_history([])
        lr.books = make_books([
            [f"B00{i}", f"Book {i}", "Author", "fiction", 2, 100 - i] for i in range(5)
        ])  # N = 5 candidates

    def test_lower_boundary_top_n_zero(self):
        """Lower boundary: top_n = 0 -> returns 0 rows."""
        result = lr.recommend("U999", top_n=0)
        self.assertEqual(len(result), 0)

    def test_just_above_lower_boundary_top_n_one(self):
        """Just above lower boundary: top_n = 1 -> returns 1 row."""
        result = lr.recommend("U999", top_n=1)
        self.assertEqual(len(result), 1)

    def test_equal_to_n_returns_all(self):
        """top_n exactly equal to N (5 candidates) -> returns all 5."""
        result = lr.recommend("U999", top_n=5)
        self.assertEqual(len(result), 5)

    def test_above_n_still_capped_at_n(self):
        """top_n greater than N -> still only returns N rows, not an error."""
        result = lr.recommend("U999", top_n=100)
        self.assertEqual(len(result), 5)

    def test_finding_negative_top_n_does_not_error(self):
        """FINDING: top_n = -1 does not raise an error and does not return 0
        rows either. Because of how pandas .head(n) works, a negative n means
        "all rows except the last n", so this returns N-1 = 4 rows instead of
        failing loudly. Documented here so the team knows to validate top_n
        before passing it in, if it will ever come from outside input."""
        result = lr.recommend("U999", top_n=-1)
        self.assertEqual(len(result), 4)  # NOT 0, and NOT an error


if __name__ == "__main__":
    unittest.main(verbosity=2)
