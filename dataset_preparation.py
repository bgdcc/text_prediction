def detect_first_uppercase(line):
    newline = ""
    for letter_id in range(1, len(line)):
        if line[letter_id].isupper():
            newline+=line[letter_id:]
            break

    return newline

def finish_the_line(line):
    newline = ""
    newerline = ''
    for letter_id in range(len(line)):
        if line[letter_id] == "+" and line[letter_id + 1] == " ":
            newline += line[letter_id + 2:]
            newerline += newline.replace("\n", "").replace(".", "").replace("\"", "").replace("!", "").replace("?", "")

    return newerline

def prepare_the_file(modified_lines):
    with open('modified_lines.txt', 'w') as f:
        for item in modified_lines:
            f.write(f"{item}\n")

def main():
    file = open("movie_lines.txt", "r", encoding = "latin1")

    lines = []
    new_lines = []
    modified_lines = []

    for line in file:
        lines.append(line)

    for line in lines:
        new_lines.append(detect_first_uppercase(line))

    for line in new_lines:
        modified_lines.append(finish_the_line(line))

    prepare_the_file(modified_lines)    

if __name__ == "__main__":
    main()