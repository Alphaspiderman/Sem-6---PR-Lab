import re

with open("text.txt", "r") as fp:
    txt = fp.read()


def match_length():
    n = int(input("Enter word length: "))
    match_n_letters = r"\W([a-zA-Z]{%s})\W"
    n_letters = re.findall(match_n_letters % n, txt)

    print(f"Words of having length of {n}:")
    print(n_letters)


def match_starting():
    pattern = input("Enter pattern to match: ")
    matched = re.findall(pattern, txt)
    print(f"Found {len(matched)} occurrences")


while True:
    print("1) Match Word Length")
    print("2) Count Pattern Occurrences")
    print("3) Quit")
    choice = int(input("Enter a Choice: "))

    if choice == 1:
        match_length()
    elif choice == 2:
        match_starting()
    elif choice == 3:
        break
    else:
        print("Invalid Choice")
