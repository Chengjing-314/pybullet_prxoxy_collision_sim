import threading
import time


def test(pc):
    global flag
    while not flag:
        print(pc)


def main():
    global flag
    flag = False
    for i in range(10):
        flag = False
        t = threading.Thread(target=test, args=(i,))
        t.start()
        time.sleep(3)
        flag = True
        t.join()


if __name__ == "__main__":
    main()

