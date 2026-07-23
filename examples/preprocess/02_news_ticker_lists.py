"""Extract ticker hints from a shared exchange-prefix list."""

from quantmind.preprocess import extract_exchange_ticker_hints

text = "Example issuer (NYSE: EVEX, EVEXW; B3: EVEB31) announced results."

for hint in extract_exchange_ticker_hints(text):
    print(hint)
