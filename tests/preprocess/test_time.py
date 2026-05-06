"""Tests for preprocess.time — UTC + parse + business-day helpers."""

import unittest
from datetime import date, datetime, timedelta, timezone

from quantmind.preprocess.time import (
    business_days_between,
    parse_filing_date,
    to_utc,
)


class ToUtcTests(unittest.TestCase):
    def test_naive_treated_as_utc(self):
        naive = datetime(2024, 4, 15, 10, 30)
        result = to_utc(naive)
        self.assertEqual(result.tzinfo, timezone.utc)
        self.assertEqual(result.hour, 10)

    def test_aware_converted_to_utc(self):
        eastern = timezone(timedelta(hours=-5))
        aware = datetime(2024, 4, 15, 10, 30, tzinfo=eastern)
        result = to_utc(aware)
        self.assertEqual(result.tzinfo, timezone.utc)
        self.assertEqual(result.hour, 15)


class ParseFilingDateTests(unittest.TestCase):
    def test_iso_with_z(self):
        result = parse_filing_date("2024-04-15T10:30:00Z")
        self.assertEqual(result.tzinfo, timezone.utc)
        self.assertEqual(result.year, 2024)
        self.assertEqual(result.hour, 10)

    def test_iso_with_offset(self):
        result = parse_filing_date("2024-04-15T10:30:00-04:00")
        self.assertEqual(result.tzinfo, timezone.utc)
        self.assertEqual(result.hour, 14)

    def test_iso_microseconds(self):
        result = parse_filing_date("2024-04-15T10:30:00.123456Z")
        self.assertEqual(result.microsecond, 123456)

    def test_date_only(self):
        result = parse_filing_date("2024-04-15")
        self.assertEqual(result.year, 2024)
        self.assertEqual(result.hour, 0)
        self.assertEqual(result.tzinfo, timezone.utc)

    def test_journal_long_form(self):
        result = parse_filing_date("Apr 15, 2024")
        self.assertEqual(result.year, 2024)
        self.assertEqual(result.month, 4)
        self.assertEqual(result.day, 15)

    def test_european_long_form(self):
        result = parse_filing_date("15 April 2024")
        self.assertEqual(result.month, 4)
        self.assertEqual(result.day, 15)

    def test_slash_separator(self):
        result = parse_filing_date("2024/04/15")
        self.assertEqual(result.year, 2024)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            parse_filing_date("   ")

    def test_unrecognised_raises(self):
        with self.assertRaises(ValueError):
            parse_filing_date("not a date")


class BusinessDaysBetweenTests(unittest.TestCase):
    def test_same_day_weekday(self):
        d = date(2024, 4, 15)  # Monday
        self.assertEqual(business_days_between(d, d), 1)

    def test_same_day_weekend(self):
        d = date(2024, 4, 13)  # Saturday
        self.assertEqual(business_days_between(d, d), 0)

    def test_full_workweek(self):
        mon = date(2024, 4, 15)
        fri = date(2024, 4, 19)
        self.assertEqual(business_days_between(mon, fri), 5)

    def test_across_weekend(self):
        fri = date(2024, 4, 19)
        mon = date(2024, 4, 22)
        self.assertEqual(business_days_between(fri, mon), 2)

    def test_two_full_weeks(self):
        mon1 = date(2024, 4, 15)
        fri2 = date(2024, 4, 26)
        self.assertEqual(business_days_between(mon1, fri2), 10)

    def test_direction_insensitive(self):
        a = date(2024, 4, 15)
        b = date(2024, 4, 22)
        self.assertEqual(
            business_days_between(a, b), business_days_between(b, a)
        )

    def test_only_saturdays_returns_zero(self):
        sat1 = date(2024, 4, 13)
        sat2 = date(2024, 4, 20)
        self.assertEqual(business_days_between(sat1, sat2), 5)
