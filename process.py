import re
import json
import jsonlines as jl


def clean(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=<>]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def process_semeval(input_file, output_file, relation_file=None):
    relation_set = set()
    lines = open(input_file, 'r', encoding='utf=8').readlines()
    with jl.open(output_file, 'w') as f:
        for index in range(0, len(lines), 4):
            line = lines[index].strip().split('\t')[1][1:-1]
            line = line.replace("<e1>", " <e1> ")
            line = line.replace("<e2>", " <e2> ")
            line = line.replace("</e1>", " </e1> ")
            line = line.replace("</e2>", " </e2> ")

            line = clean(line)
            token = line.split(' ')
            e1_start_index, e1_end_index = token.index('<e1>'), token.index('</e1>')
            e2_start_index, e2_end_index = token.index('<e2>'), token.index('</e2>')

            relation = lines[index + 1].strip()

            f.write({
                'token': token,
                'head': [],
                'tail': [],
                'relation': relation
            })

    with open(relation_file, 'w', encoding='utf-8') as f:
        for relation in relation_set:
            f.write(relation + '\n')


def process_semeval_dual_relation(input_file, output_file, relation_file=None):
    relation_set = set()
    lines = open(input_file, 'r', encoding='utf-8').readlines()
    with jl.open(output_file, 'w') as f:
        for index in range(0, len(lines), 4):
            line = lines[index].strip().split('\t')[1][1:-1]
            line = line.replace("<e1>", " <e1> ")
            line = line.replace("<e2>", " <e2> ")
            line = line.replace("</e1>", " </e1> ")
            line = line.replace("</e2>", " </e2> ")

            line = clean(line)
            token = line.split(' ')
            e1_start_index, e1_end_index = token.index('<e1>'), token.index('</e1>')
            e2_start_index, e2_end_index = token.index('<e2>'), token.index('</e2>')

            for entity_index in sorted([e1_start_index, e1_end_index, e2_start_index, e2_end_index], reverse=True):
                token.pop(entity_index)

            if e1_start_index < e2_start_index:
                e1_start_index = e1_start_index
                e1_end_index = e1_end_index - 1
                e2_start_index = e2_start_index - 2
                e2_end_index = e2_end_index - 3
            else:
                e2_start_index = e2_start_index
                e2_end_index = e2_end_index - 1
                e1_start_index = e1_start_index - 2
                e1_end_index = e1_end_index - 3

            relation = lines[index + 1].strip()
            if relation != 'Other':
                relation = relation.split('(')[0]

            relation_set.add(relation)

            f.write({
                'token': token,
                'head': [" ".join(token[e1_start_index:e1_end_index]), list(range(e1_start_index, e1_end_index))],
                'tail': [" ".join(token[e2_start_index:e2_end_index]), list(range(e2_start_index, e2_end_index))],
                'relation': relation
            })

    if relation_file is not None:
        with open(relation_file, 'w', encoding='utf-8') as f:
            for relation in relation_set:
                f.write(relation + '\n')


def process_tacred():
    pass


def process_embedding(input_file, output_file, embedding_dim=300):
    with jl.open(output_file, 'w') as f:
        for line in open(input_file, 'r', encoding='utf-8'):
            line = line.strip().split(' ')
            token = line[:-embedding_dim]
            vector = [float(v) for v in line[-embedding_dim:]]
            f.write({
                'token': " ".join(token),
                'vec': vector
            })


if __name__ == '__main__':
    process_semeval_dual_relation('./raw_data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT', './data/semeval/train.jl', './data/semeval/relation.txt')
    process_semeval_dual_relation('./raw_data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT', './data/semeval/test.jl')
    # process_embedding('./raw_data/embedding/glove.6B.50d.txt', './data/embedding/glove.6B.50d.jl', embedding_dim=50)
