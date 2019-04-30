lucky_numbers = [1, 2, 3, 8, 9,]
friends = ["naitik ", "keyur", "parth", "motu", "nidhish", "rahul" ]
friends.extend(lucky_numbers)
friends.append("creed")
friends.insert(1 , "kelly")
friends.remove("motu")
friends.pop()
print(friends)
friends2 = friends.copy
print(friends2)