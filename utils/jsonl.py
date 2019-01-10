import json

def loads(fin):
    datas = []
    for line in fin:
        data = json.loads(line)
        datas.append(data)
    fin.close()
    return datas

def dumps(datas, fout):
    for data in datas:
        print(
            json.dumps(
                data, 
                ensure_ascii=False
            ),
            file=fout,
        )
    fout.close()