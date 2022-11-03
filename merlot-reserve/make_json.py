import os
import json
import re
import subprocess
import sys
import random
import os
print(os.environ["PATH"])


test = "test"
sys.path.append("/home/ec2-user")
os.environ["PATH"] += ":/usr/local/bin/ffmpeg"

import socialiq_std_folds

def make_json_for(vids, file_name):
    for vid in vids:
        vid_name = vid + "_trimmed-out.mp4"
        vid_length = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=duration', '-of', 'default=noprint_wrappers=1:nokey=1', os.path.join("/data/raw/vision/raw", vid_name)])
        
        vid_filename = os.path.join("/data/raw/qa", vid + "_trimmed.txt")
        vid_file = open(vid_filename, "r") 

        file_end = False

        while not file_end:
            line = vid_file.readline()
            if not line:
                vid_file.close()
                break
            if bool(re.match(r"q\d+:*(.)", line)):
                vid_dict = {}
                vid_dict["vid_name"] = vid
                vid_dict["ts"] = "0.00-" + vid_length.decode("utf-8").strip() # timestamp corresponding to question

                # question
                question_num = line.split(':')[0]
                str_list = line.split(':')[1:]
                question = ':'.join(str_list).strip()
                vid_dict['q'] = question
                vid_dict["qid"] = vid + "_" + question_num
                # if vid in ["C08WmKiwcSs", "n5_HdNzf03Q", "y1Y02_oZP8U", "SjrMprYa608", "D2g3gTRkv0U"]:
                #     continue
                # if vid_dict["qid"] in ["58DqoE56OWc_q2", "T8JwNZBJ_wI_q7", "2XFVnzr4Vho_q1", "eDqEcrIRxgQ_q2", "eDqEcrIRxgQ_q3", "eDqEcrIRxgQ_q4", "eDqEcrIRxgQ_q5", "Jp_KHLvQcuw_q3", "Jp_KHLvQcuw_q6", "Jp_KHLvQcuw_q11", "bMuoPr5-Yt4_q3", "gbVOyKifrAo_q1", "gbVOyKifrAo_q2", "gbVOyKifrAo_q6", "srWtQnseRyE_q1", "srWtQnseRyE_q2", "srWtQnseRyE_q3", "q45sJ2n2XPg_q1", "q45sJ2n2XPg_q5", "q45sJ2n2XPg_q6", "_UJNNySGM6Q_q2", "_UJNNySGM6Q_q3", "_UJNNySGM6Q_q4", "VP4rHzYyuL0_q2", "VP4rHzYyuL0_q4", "VP4rHzYyuL0_q6", "N-6zVmVuTs0_q2", "N-6zVmVuTs0_q6", "N-6zVmVuTs0_q7",
                #                         "gBs-CkxGXy8_q2", "gBs-CkxGXy8_q3", "gBs-CkxGXy8_q6", "gBs-CkxGXy8_q7", "gBs-CkxGXy8_q10", "gBs-CkxGXy8_q11", "E4MUXs4IHtY_q4", "ZuYTtKZUkJc_q1", "ZuYTtKZUkJc_q2", "j1CTHVQ8Z3k_q1", "j1CTHVQ8Z3k_q3", "j1CTHVQ8Z3k_q4", "j1CTHVQ8Z3k_q5", "j1CTHVQ8Z3k_q6", "j1CTHVQ8Z3k_q7", "erOpqmubBL4_q1", "H0Qdz8bSkv0_q2", "H0Qdz8bSkv0_q3", "H0Qdz8bSkv0_q4", "mpHoYhIFKNI_q1", "aqGNOsZFdBU_q1", "aqGNOsZFdBU_q5", "FositxHjuUk_q1"]:
                #     continue
                answer_num = 0
                correct_num = 0
                incorrect_num = 0
                all_correct_ans = []
                all_incorrect_ans = []
                while True:
                    pos = vid_file.tell()
                    next_line = vid_file.readline()
                    if not next_line:
                        vid_file.close()
                        file_end = True
                        break
                    if bool(re.match(r"a:*(.)", next_line)):
                        # correct answer
                        ans_str_list = next_line.split(':')[1:]
                        answer = ':'.join(ans_str_list).strip()
                        if correct_num < 4 and answer not in all_correct_ans:
                            answer_num += 1
                            correct_num += 1
                            all_correct_ans.append(answer)
                    elif bool(re.match(r"i:*(.)", next_line)):
                        # incorrect answer
                        ans_str_list = next_line.split(':')[1:]
                        answer = ':'.join(ans_str_list).strip()
                        if incorrect_num < 3 and answer not in all_incorrect_ans:
                            answer_num += 1
                            incorrect_num += 1
                            all_incorrect_ans.append(answer)
                    else:
                        # question
                        # if len(all_incorrect_ans) != 3:
                        #     print(vid_dict['qid'])
                            # vid_file.seek(pos)
                            # break
                        # for ans in all_correct_ans:
                        #     all_answers = []
                        #     all_answers.extend(all_incorrect_ans)
                        #     all_answers.append(ans)
                        #     random.shuffle(all_answers)
                        #     for j in range(4):
                        #         curr_ans = all_answers.pop(0)
                        #         vid_dict["a"+str(j)] = curr_ans
                        #         if curr_ans in all_correct_ans:
                        #             vid_dict['answer_idx'] = j
                        #         file_name.write(json.dumps(vid_dict) + "\n")
                        product = [[a, b] for a in all_correct_ans for b in all_incorrect_ans]
                        for answers in product: 
                            random.shuffle(answers)                           
                            vid_dict["a0"] = answers[0]
                            vid_dict["a1"] = answers[1]
                            if answers[0] in all_correct_ans:
                                vid_dict['answer_idx'] = 0
                            else:
                                vid_dict['answer_idx'] = 1
                            file_name.write(json.dumps(vid_dict) + "\n")
                            # print(vid_dict)
                        vid_file.seek(pos)
                        break    

train_vids = socialiq_std_folds.standard_train_fold
val_vids = socialiq_std_folds.standard_valid_fold

all_vids = os.listdir("/data/raw/vision/raw")

train_file = open("/data/raw/siq_train.jsonl", "w")
val_file = open("/data/raw/siq_val.jsonl", "w")

make_json_for(train_vids, train_file)
make_json_for(val_vids, val_file)
