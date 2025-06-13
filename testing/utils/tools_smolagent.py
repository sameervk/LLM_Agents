from smolagents import tool

@tool
def calculator(a: float, b: float) -> float:
    """
    Multiply two integers
    Args:
        a: input to the function
        b: input to the function

    """
    return a*b

print(calculator.to_dict())
