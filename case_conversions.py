text = "python PROGRAMMING"
print(text.upper())

text = "python PROGRAMMING"
print(text.lower())

#  "968-Maria, ( D@t@ Engineer ) ;; 27y " 
# name: maria | role: data engineer | age 27

text = "968-Maria, ( D@t@ Engineer ) ;; 27y"
print(f"name: {text[4:9]} | role : {text[13:27].replace("(","").replace(")","").replace("@","a").replace(";","") } | age:{text[-3:-1]}")