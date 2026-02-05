"""
Test script to verify Kraken asset name mapping works correctly.
Run this BEFORE deploying to make sure XRP and other assets map correctly.
"""

# Simulated Kraken balance response (from your actual logs)
SIMULATED_KRAKEN_BALANCE = {
    'ADA': 0.00156207,
    'AVAX': 0.0,
    'BABY': 0.00672,
    'DOT': 0.0003516479,
    'LINK': 0.0,
    'SOL': 0.3002368814,
    'USD.HOLD': 0.0,
    'ETH': 0.0095798332,
    'LTC': 0.37972892,
    'BTC': 0.00079667,
    'DOGE': 0.0,
    'XRP': 9.39189251,
    'USD': 19.36
}

WATCHLIST = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "LTC/USD"]

# Asset mapping (same as in bot)
KRAKEN_ASSET_MAP = {
    "XXBT": "BTC",
    "XBT": "BTC",
    "XETH": "ETH",
    "ZUSD": "USD",
    "XXRP": "XRP",
}

def map_asset(asset: str) -> str:
    """Map Kraken asset name to clean ticker symbol."""

    # Use explicit mapping first
    if asset in KRAKEN_ASSET_MAP:
        return KRAKEN_ASSET_MAP[asset]

    # Smart logic for unmapped assets
    if asset.startswith(("X", "Z")) and len(asset) > 3:
        test_asset = asset[1:]
        if 3 <= len(test_asset) <= 4:
            return test_asset
        else:
            return asset
    else:
        return asset

# Test the mapping
print("="*60)
print("Testing Kraken Asset Name Mapping")
print("="*60)

for asset, amount in SIMULATED_KRAKEN_BALANCE.items():
    if asset in ["USD", "ZUSD"] or amount <= 0:
        print(f"⏭️  '{asset}' → SKIPPED (USD or zero balance)")
        continue

    clean_asset = map_asset(asset)
    pair = f"{clean_asset}/USD"

    in_watchlist = "✅" if pair in WATCHLIST else "❌"

    print(f"{in_watchlist}  '{asset}' → '{clean_asset}' → '{pair}' | Amount: {amount:.8f}")

print("\n" + "="*60)
print("Expected Results:")
print("="*60)
print("✅ BTC/USD should appear")
print("✅ ETH/USD should appear")
print("✅ SOL/USD should appear")
print("✅ XRP/USD should appear (NOT RP/USD!)")
print("✅ LTC/USD should appear")
print("❌ ADA/USD should be skipped (not in watchlist)")
print("❌ Other coins should be skipped")

# Critical test: XRP specifically
print("\n" + "="*60)
print("CRITICAL TEST: XRP Mapping")
print("="*60)

test_cases = [
    ("XRP", "XRP", "XRP/USD"),      # Modern format
    ("XXRP", "XRP", "XRP/USD"),     # Legacy format
    ("XRPUSD", "RPUSD", "RPUSD/USD"),  # Edge case (shouldn't happen but test anyway)
]

for kraken_name, expected_clean, expected_pair in test_cases:
    actual_clean = map_asset(kraken_name)
    actual_pair = f"{actual_clean}/USD"

    if actual_clean == expected_clean and actual_pair == expected_pair:
        print(f"✅ '{kraken_name}' → '{actual_clean}' → '{actual_pair}' (CORRECT)")
    else:
        print(f"❌ '{kraken_name}' → '{actual_clean}' → '{actual_pair}' (EXPECTED: '{expected_clean}' → '{expected_pair}')")

print("\n" + "="*60)
print("If all ✅ appear for your coins, the mapping is CORRECT!")
print("="*60)