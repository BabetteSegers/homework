def divider(word):
    returned_word = ''
    if len(word) % 3 == 0 and len(word) % 5 == 0:
        returned_word = 'three&five'
    elif len(word) % 3 == 0:
        returned_word = 'three'
    elif len(word) % 5 == 0:
        returned_word = 'five'
    else:
        returned_word = word
        
    return returned_word