
# fin1 = open('../train/seq.in')
# fin2 = open('../test/seq.in')

fin1 = open('snips/train/seq.in')
fin2 = open('snips/test/seq.in')

wc_dct = {}
for line in fin1:
    line = line.strip()
    line_array = line.split(' ')
    for word in line_array:
        if word not in wc_dct:
            wc_dct[word] = 1
        else:
            wc_dct[word] += 1

oov_dct = {}
for line in fin2:
    line = line.strip()
    line_array = line.split(' ')
    for word in line_array:
        if word not in wc_dct:
            if word not in oov_dct:
                oov_dct[word] = 1
            else:
                oov_dct[word] += 1

print(oov_dct)
print(len(oov_dct))
print(len(wc_dct))