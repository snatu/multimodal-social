import getopt
import sys

def produce_qa(argv):
    try:
      opts, args = getopt.getopt(argv,"p:")
    except getopt.GetoptError:
        print('human_in_box.py -p <path to qa text file>')
        sys.exit(2)
    path = ''
    for opt, arg in opts:
      if opt in ("-p"):
         path = arg

    QAs = {}
    with open(path, "r") as f:
        question = None
        answers = []
        for line in f.readlines():
            if line[0] == 'q':
                if question is not None:
                    QAs[question] = answers
                answers = []
                question = line.split(":")[1]
            else:
                answers.append(line.split(":")[1])
    
    write_path = path[:-4] + "_processed.txt"
    with open(write_path, "w") as f:
        for question in QAs.keys():
            f.write(question + '\n')
            i = 1
            for answer in QAs[question]:
                f.write(str(i) + ")" + answer + '')
                i += 1

if __name__ == "__main__":
    produce_qa(sys.argv[1:])