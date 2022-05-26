def calculate_priori(y: list, filter_by: int):
    count = len(list(filter(lambda i: i == filter_by, y)))
    return count, count / len(y)


y = [0, 1, 0, 1, 0, 1]

print(calculate_priori(y, 1))
print(calculate_priori(y, 0))
