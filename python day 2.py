r = float(input("Enter the radius of the circle: "))
aoc = 2 * 3.14 * r * r
print("Area of the circle is:", aoc)

# Circumference of circle
r = float(input("Enter the radius of the circle: "))
circumference = 2 * 3.14 * r
print("Circumference of the circle is:", circumference)

# Area of rectangle
length = float(input("Enter the length of the rectangle: "))
width = float(input("Enter the width of the rectangle: "))
area_rect = length * width
print("Area of the rectangle is:", area_rect)

# Volume of sphere
r = float(input("Enter the radius of the sphere: "))
volume_sphere = (4/3) * 3.14 * r * r * r
print("Volume of the sphere is:", volume_sphere)

# Area of sphere
r = float(input("Enter the radius of the sphere: "))
area_sphere = 4 * 3.14 * r * r
print("Area of the sphere is:", area_sphere)

# Area of square
side = float(input("Enter the side of the square: "))
area_square = side * side
print("Area of the square is:", area_square)

# Area of right-angle triangle
base = float(input("Enter the base of the right-angle triangle: "))
height = float(input("Enter the height of the right-angle triangle: "))
area_right_triangle = 0.5 * base * height
print("Area of the right-angle triangle is:", area_right_triangle)

# Area of equilateral triangle
side = float(input("Enter the side of the equilateral triangle: "))
area_equilateral = (1.732 / 4) * side * side
print("Area of the equilateral triangle is:", area_equilateral)

# Perimeter of rectangle
length = float(input("Enter the length of the rectangle: "))
width = float(input("Enter the width of the rectangle: "))
perimeter_rect = 2 * (length + width)
print("Perimeter of the rectangle is:", perimeter_rect)

# Area of triangle
base = float(input("Enter the base of the triangle: "))
height = float(input("Enter the height of the triangle: "))
area_triangle = 0.5 * base * height
print("Area of the triangle is:", area_triangle)

# Simple interest
principal = float(input("Enter the principal amount: "))
rate = float(input("Enter the rate of interest: "))
time = float(input("Enter the time in years: "))
simple_interest = (principal * rate * time) / 100
print("Simple interest is:", simple_interest)

# Compound interest
principal = float(input("Enter the principal amount: "))
rate = float(input("Enter the rate of interest: "))
time = float(input("Enter the time in years: "))
n = float(input("Enter the number of times interest is compounded per year: "))
compound_interest = principal * (1 + rate/(100 * n))**(n * time) - principal
print("Compound interest is:", compound_interest)

# Convert days into years
days = float(input("Enter the number of days: "))
years = days / 365
print("Years:", years)

# Fahrenheit to Celsius
fahrenheit = float(input("Enter the temperature in Fahrenheit: "))
celsius = (fahrenheit - 32) * 5 / 9
print("Temperature in Celsius is:", celsius)
