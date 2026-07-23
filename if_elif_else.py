score = int(input("Enter your score: "))
submitted_project = input("Did you submit the project? (yes/no): ").lower() == "yes"
if score >= 90 and submitted_project:
    print("You got an A!")
elif score >= 80:
    print ("you got a B!")
elif score >= 70:
    print("you got a C!")
elif score >= 60:
    print("you got a D!")
else :
    print("You got an F!")
