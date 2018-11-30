# Collections: tuples, lists, dictionaries
# Operations

roster = ['Nimish', 'John', 'Kate', 'Sarah', 'Kevin']

inventory = {'apple': 5, 'knife': 1, 'shoes': 2}
print(inventory['apple'])

first_name = ('Nimish', "Narang")
last_name = ('John', 'Smith')

first_name = first_name + last_name;
# del first_name
print(len(first_name))
print(min(first_name))
print(max(first_name))

roster[0] = "Lorence"
print(roster)
print(type(roster))
del roster[0]
print(roster)
roster.clear()
print(roster)

