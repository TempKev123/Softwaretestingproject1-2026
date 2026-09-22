"""Unit tests for libraryrec_minimal.

The module under test is unmodified. Expected results state what each
function should return; where the program returns something else the
case is recorded as a failure.

Run with:  python -m unittest -v test_libraryrec
"""

import io
import unittest
from contextlib import redirect_stdout

import pandas as pd

import libraryrec_minimal as lib

# Top three entries of U001's real tag profile, used as a short,
# hand-checkable preference list for the calculate_score cases.
PREFS = [("fantasy", 3), ("adventure", 3), ("fiction", 2)]

U001_BORROWED = {"B003", "B006", "B009", "B016", "B029", "B030"}


class DataLoaded(unittest.TestCase):
    """Reloads the real data before each test and restores it after."""

    def setUp(self):
        self._saved = (lib.books, lib.borrow_history)
        lib.load_data("dummydata")

    def tearDown(self):
        lib.books, lib.borrow_history = self._saved


# ===========================================================================
# calculate_score
# ===========================================================================

class TestCalculateScore(unittest.TestCase):

    def test_case_1_single_matching_tag(self):
        self.assertEqual(lib.calculate_score("fantasy", PREFS), 3)

    def test_case_2_multiple_matching_tags_summed(self):
        self.assertEqual(lib.calculate_score("fantasy,adventure", PREFS), 6)

    def test_case_3_unmatched_tag_scores_zero(self):
        self.assertEqual(lib.calculate_score("biography", PREFS), 0)

    def test_case_4_empty_preferences_scores_zero(self):
        self.assertEqual(lib.calculate_score("fantasy,adventure", []), 0)

    def test_case_5_matching_is_case_insensitive(self):
        self.assertEqual(lib.calculate_score("Fantasy", PREFS), 3)


# ===========================================================================
# findTags
# ===========================================================================

class TestFindTags(DataLoaded):

    def test_case_1_tag_profile_for_known_user(self):
        tags, _ = lib.findTags("U001")
        self.assertEqual(tags[:4],
                         [("fantasy", 3), ("adventure", 3),
                          ("fiction", 2), ("children", 2)])

    def test_case_2_borrowed_ids_for_known_user(self):
        _, borrowed = lib.findTags("U001")
        self.assertEqual(borrowed, U001_BORROWED)

    def test_case_3_unknown_user_returns_empty(self):
        self.assertEqual(lib.findTags("U999"), ([], set()))

    def test_case_4_profile_ordered_by_descending_count(self):
        tags, _ = lib.findTags("U001")
        counts = [c for _, c in tags]
        self.assertEqual(counts, sorted(counts, reverse=True))


# ===========================================================================
# recommend
# ===========================================================================

class TestRecommend(DataLoaded):

    def test_case_1_excludes_already_borrowed_books(self):
        result = lib.recommend("U001", 5)
        self.assertEqual(set(result["book_id"]) & U001_BORROWED, set())

    def test_case_2_excludes_unavailable_books(self):
        result = lib.recommend("U001", 30)
        self.assertFalse((result["available"] == 0).any())

    def test_case_3_ranked_by_descending_score(self):
        scores = list(lib.recommend("U001", 5)["score"])
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_case_4_respects_top_n(self):
        self.assertEqual(len(lib.recommend("U001", 3)), 3)

    def test_case_5_unknown_user_ranked_by_popularity(self):
        result = lib.recommend("U999", 5)
        pops = list(result["popularity"])
        self.assertTrue((result["score"] == 0).all())
        self.assertEqual(pops, sorted(pops, reverse=True))

    def test_case_6_negative_top_n_rejected(self):
        with self.assertRaises(ValueError):
            lib.recommend("U001", -1)


# ===========================================================================
# test
# ===========================================================================

class TestTest(DataLoaded):

    def test_case_1_prints_a_block_for_each_user(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            lib.test()
        output = buf.getvalue()
        for user in ["U001", "U002", "U003", "U004", "U005"]:
            self.assertIn(f"Recommendations for {user}:", output)

    def test_case_2_runs_when_data_is_absent(self):
        lib.books = pd.DataFrame()
        lib.borrow_history = pd.DataFrame()
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                lib.test()
        except KeyError as exc:
            self.fail(f"test() raised KeyError({exc}) instead of reporting "
                      f"that no recommendations could be produced")


if __name__ == "__main__":
    unittest.main(verbosity=2)
