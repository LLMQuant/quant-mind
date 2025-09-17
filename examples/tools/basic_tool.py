"""Basic example of creating and using a QuantMind tool."""

from quantmind.tools import tool, validate_tool_arguments


@tool
def calculate_position_value(
    price: float, quantity: float, side: str = "long"
) -> float:
    """Compute the signed notional value for a position.

    Args:
        price (float): Latest unit price in dollars.
        quantity (float): Position size in units.
        side (str): Trading direction (choices: ["long", "short"])

    Returns:
        float: Signed notional value for the position.
    """
    direction = 1 if side == "long" else -1
    return direction * price * quantity


def main():
    """Run the tool, validate inputs, and display metadata."""
    payload = {"price": 420.5, "quantity": 2, "side": "long"}
    validate_tool_arguments(calculate_position_value, payload)
    result = calculate_position_value(**payload)
    print(f"Position value: {result}")
    print("Tool inputs:")
    for name, schema in calculate_position_value.inputs.items():
        print(f"  {name}: {schema}")


if __name__ == "__main__":
    main()
