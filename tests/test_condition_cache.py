import unittest
from forward_sta import condition


class ConditionCache(unittest.TestCase):
    def test_boolean_normalization_is_unchanged_and_results_are_immutable(self):
        condition.cache_clear()
        for expression in [None,'','A','!A','A&B','A+B','(A&B)+(!A&C)']:
            expected=condition.__wrapped__(expression)
            self.assertEqual(condition(expression),expected)
            self.assertIs(condition(expression),condition(expression))
        self.assertGreater(condition.cache_info().hits,0)
        self.assertEqual(condition.cache_info().maxsize,4096)
        for function in [condition,condition.__wrapped__]:
            with self.assertRaises(ValueError):function('A B')


if __name__=='__main__':unittest.main()
