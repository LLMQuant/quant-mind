"""Basic example of creating and using a QuantMind tool."""

import asyncio

import aiohttp

from quantmind.tools import tool


@tool
async def get_btc_usdt_price() -> float:
    """Fetch the latest BTC/USDT price from Binance API."""
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
            return float(data["price"])


async def main():
    """Main function to run the example."""
    price = await get_btc_usdt_price.run()
    print(f"BTC/USDT price: {price}")


if __name__ == "__main__":
    asyncio.run(main())
