def detect_first_uppercase(line):
    newline = ""
    for letter_id in range(1, len(line)):
        if line[letter_id].isupper():
            newline+=line[letter_id:]
            break

    return newline

file = open("movie_lines.txt", "r", encoding = "latin1")

lines = []
new_lines = []

for line in file:
    lines.append(line)

for line in lines:
    new_lines.append(detect_first_uppercase(line))

print(new_lines)    