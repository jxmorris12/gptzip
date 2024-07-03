def float2bin(number: float, places: int=50) -> str:
    # originally from
    #  https://github.com/Jonny-exe/arithmetic-coding-compression/blob/master/src/python/helpers.py
    number = number
    rest = 0
    result = ""
    b = 1
    i = 1
    while i < places:
        b = 2 ** -i
        if b + rest <= number:
            result += "1"
            rest += b
        else:
            result += "0"
        i += 1
    return result


def bin2float(number: str) -> float:
    # TODO: make this work with not only intervals from 0 to 1
    result = 0
    number = number[number.find(".") + 1 :]
    for i in range(len(number)):
        c = number[i]
        if c == "1":
            result += 2 ** -(i + 1)
    return result