def revword(word):
            word = word.lower()[::-1]
            return word


def countword():
    fname = open("text.txt")
    rep_count = 1
    word_counter=0
    for line in fname:
        if rep_count==1:
            word=line.lower().rstrip()
            rep_count+=1
            word_counter+=1
        else:
            line=line.split()
            for fix in line:
                fix_word=revword(fix)
                if word==fix_word:
                    word_counter+=1
    return word_counter

print(countword())