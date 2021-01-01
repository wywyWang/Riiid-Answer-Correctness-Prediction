if __name__ == "__main__":
    min_part, min_tag, max_part, max_tag = 100, 100, -1, -1
    max_tags_len = 0
    with open('data/questions.csv', 'r') as f:
        next(f)
        for row in f.readlines():
            data = row.strip().split(',')
            part, tags = int(data[-2]), data[-1].split(' ')
            if part > max_part:
                max_part = part
            if part < min_part:
                min_part = part
            tags_len = 0
            for tag in tags:
                if tag == '':
                    continue
                tags_len += 1
                tag = int(tag)
                if tag > max_tag:
                    max_tag = tag
                if tag < min_tag:
                    min_tag = tag
            if tags_len > max_tags_len:
                max_tags_len = tags_len

    print(f'Min part: {min_part}, Max part: {max_part}')
    print(f'Min tag: {min_tag}, Max tag: {max_tag}')
    print(f'Max tags len: {max_tags_len}')