import os

def calc_bleu_score(predictions, references, log_dir):
    pred_file = os.path.join(log_dir, 'pred.txt')
    ref_file = os.path.join(log_dir, 'ref.txt')

    with open(ref_file, 'w', encoding='utf-8') as f:
        for s in references:
            print(s, file=f,)
    with open(pred_file, 'w', encoding='utf-8') as f:
        for s in predictions:
            print(s, file=f,)

    temp = os.path.join(log_dir, "result.txt")
    
    command = "perl metrics/multi-bleu.perl " + ref_file + "<" + pred_file + "> " + temp
    os.system(command)
    with open(temp) as ft:
        result = ft.read()
    os.remove(temp)
    print(result)
    score = float(result.split()[2][:-1])

    return score