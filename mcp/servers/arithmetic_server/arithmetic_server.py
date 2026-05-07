from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

class ArithmeticInput(BaseModel):
    a: float = Field(..., description="The first number")
    b: float = Field(..., description="The second number")

class ArithmeticOutput(BaseModel):
    result: float = Field(..., description="The result of the arithmetic operation")
    expression: str = Field(..., description="The arithmetic expression that was evaluated")

mcp = FastMCP("arithmetic_server",
              host="localhost",
              port=8000,
              stateless_http=True)

@mcp.tool("add", "Add two numbers")
async def add(input: ArithmeticInput) -> ArithmeticOutput:
    """
    Add two numbers and return the result along with the expression.

    Args:
        input (ArithmeticInput): An object containing the two numbers to be added.

    Returns:
        ArithmeticOutput: An object containing the result of the addition and the expression.
    """

    result = input.a + input.b
    expression = f"{input.a} + {input.b} = {result}"
    return ArithmeticOutput(result=result, expression=expression)

@mcp.tool("subtract", "Subtract two numbers")
async def subtract(input: ArithmeticInput) -> ArithmeticOutput:
    """
    Subtract two numbers and return the result along with the expression.

    Args:
        input (ArithmeticInput): An object containing the two numbers to be subtracted.

    Returns:
        ArithmeticOutput: An object containing the result of the subtraction and the expression.
    """

    result = input.a - input.b
    expression = f"{input.a} - {input.b} = {result}"
    return ArithmeticOutput(result=result, expression=expression)

@mcp.tool("multiply", "Multiply two numbers")
async def multiply(input: ArithmeticInput) -> ArithmeticOutput:
    """
    Multiply two numbers and return the result along with the expression.

    Args:
        input (ArithmeticInput): An object containing the two numbers to be multiplied.

    Returns:
        ArithmeticOutput: An object containing the result of the multiplication and the expression.
    """

    result = input.a * input.b
    expression = f"{input.a} * {input.b} = {result}"
    return ArithmeticOutput(result=result, expression=expression)

@mcp.tool("divide", "Divide two numbers")
async def divide(input: ArithmeticInput) -> ArithmeticOutput:
    """
    Divide two numbers and return the result along with the expression.

    Args:
        input (ArithmeticInput): An object containing the two numbers to be divided.

    Returns:
        ArithmeticOutput: An object containing the result of the division and the expression.
    """

    if input.b == 0:
        raise ValueError("Cannot divide by zero.")
    
    result = input.a / input.b
    expression = f"{input.a} / {input.b} = {result}"
    return ArithmeticOutput(result=result, expression=expression)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")