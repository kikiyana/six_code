def generate_partitions(n, k, a):
    global s
    if n > 0:
        for i in range(n, 0, -1):
            a[k] = i
            generate_partitions(n - i, k + 1, a)
    else:
        for i in range(k):
            print(f"{a[i]:5d}", end="")
        print()
        s += 1

def main():
    global s
    n = int(input())
    s = 0
    a = [0] * 10
    generate_partitions(n, 0, a)
    print(f"s={s}")

if __name__ == "__main__":
    main()
