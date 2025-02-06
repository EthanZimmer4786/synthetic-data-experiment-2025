def print_chart(frequencies, name):
    for i in range(len(frequencies)):
        frequencies[i] = round(frequencies[i] * 50)

    bars = [''] * len(frequencies)
    for i in range(10):
        for j in range(frequencies[i]):
            bars[i] = bars[i] + '|'

    print(name)
    for i in range(len(bars)):
        print(str(i) + ' ' + bars[i])