def test(age):
  assert type(age) is int, 'age 값은 정수만 가능'
  assert age > 0, 'age 값은 양수만 가능'

age = 1
test(age)

age = -10
test(age)
