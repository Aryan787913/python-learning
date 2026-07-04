# - use print() to recreate this exact output
# - you are allowed to use only one print()

# Your Learning Path:
#-> - Python Basics
#-> - Data Engineering
#-> - AI

print("""Your Learning Path:
        \t - Python Basics
        \t - Data Engineering
        \t - AI""")

# print the following three lines
# Add a vaiable to make a dynamic

# info@datawitharyan.com
# support@datawitharyan.com
# www.datawitharyan.com

username = "datawitharyan"
com = ".com"
print("info"+username+com)
print("info",username,com, sep="")
print("support"+username+com)
print("www."+username+com)

#variables 

#create 5 variables - each with different data type:
#1. Your age 
#2. Your height(with decimals)
#3. your name 
#4. Are you Student?
#5. Soemthing with no value yet

a=int(18)
b = float(170.2)
c = "AAryan"
d = bool(True)
e = None
print(a)
print(b)
print(c)
print(d)
print(e)

text = " a , a , c , c, d"
a = text.upper()
b = a.count("D")
print(b)

phone = "98-765-828"
a = phone.replace("-","")
print(a)


#replace
# +49 (176) 123-4567
phone = "+49 (176) 123-4567"
a = phone.replace("+","").replace(" ","").replace("-","").replace("(","").replace(")","")
print(a)

#string + string


fn = "aaryan"
ln = "singh"
lnn = fn +" " + ln 
print(lnn)
#f-string
print(f"my first name is {fn}, and last name is {ln}, and full name is {lnn}")
print(f"{{my first name is {fn}, and last name is {ln}, and full name is {lnn}}}")