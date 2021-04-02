import json

law2num = json.load(open('glaw2num.json', 'r'))
lawcontent = {law['title']: {tiao: chapter[tiao] for chapter in law['content'] for tiao in chapter} for law in json.load(open('good_laws.json', 'r'))}
print(len(json.load(open('good_laws.json', 'r'))))
print(len(lawcontent))
laws = []
for law in law2num:
    try:
        key = law.split('-')
        tiao = lawcontent[key[0]][key[1]]
        if type(tiao) == str:
            laws.append(tiao)
        else:
            laws.append(tiao[key[2]])
    except:
        from IPython import embed; embed()
fout = open('id2lawlist.json', 'w')
print(json.dumps(laws, ensure_ascii=False, indent=2), file=fout)
fout.close()