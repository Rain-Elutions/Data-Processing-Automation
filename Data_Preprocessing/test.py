from typing import Protocol

class HasArea(Protocol):
    def area(self) -> float:
        ...

class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

# Uncommenting the following code will raise an error
class Circle:
    def __init__(self, radius: float):
        self.radius = radius

# This will raise an error if print_area function is used
# with an instance of Circle without the HasArea protocol
# Circle does not have a .area() method.
def print_area(shape: HasArea) -> None:
    print("Area:", shape.area())

rectangle = Rectangle(width=5, height=3)
print_area(rectangle)

circle = Circle(radius=4)
print_area(circle)  # This would raise an error
