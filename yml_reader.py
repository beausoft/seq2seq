import yaml
import glob

ymls = glob.glob('C:/Users/user/Desktop/chinese/*.yml')
print(ymls)

question = []
answer = []

for i, yml_path in enumerate(ymls):
    with open(yml_path, encoding='utf8') as yml_file:
        datas = yaml.load(yml_file)
        for j, data in enumerate(datas['conversations']):
            question.append(data[0]+'\n')
            answer.append(data[1]+'\n')

with open('question.txt', encoding='gbk', mode='w') as question_file:
    question_file.writelines(question)

with open('answer.txt', encoding='gbk', mode='w') as answer_file:
    answer_file.writelines(answer)
