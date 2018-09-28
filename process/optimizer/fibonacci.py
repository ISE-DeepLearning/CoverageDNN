# 递归
def fibonacci(n):
    if n < 0:
        raise Exception('it is not what we want!')
    if n == 1 or n == 0:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


# 循环
def fibonacci_loop(n):
    if n == 1 or n == 0:
        return 1
    a = 1
    b = 1
    result = a
    for i in range(2, n + 1):
        a = a + b


if __name__ == '__main__':
    num = fibonacci(10)
    print(num)
