import os
from mimixlm import *

def eval_exam(args):
    """
    Evaluates language model performance on Chinese multiple-choice question answering.
    """

    def extract_answer(data, gen_ans):
        """
        Extracts multiple-choice answers from generated text using regex patterns.
        """
        answer_patterns = [
            r'答案[：:][ ]*([ABCD])',
            r'所以答案是([ABCD])'
            r'答案为[-]*([ABCD])',
            r'答案是[：:][ ]*([ABCD])'
            r'选择([ABCD])'
            r'选择答案([ABCD])',
            r'正确答案是([ABCD])',
            r'故选[：: ]*([ABCD])',
            r'故*应选[：: ]*([ABCD])',
            r'故*答案应选[：: ]*([ABCD])',
            r'所以选[：: ]*([ABCD])',
            r'所以应选[：: ]*([ABCD])',
            r'所以答案应选[：: ]*([ABCD])',
            r'因此选([ABCD])',
            r'^([ABCD])',
        ]
        for answer_pattern in answer_patterns:
            m = re.search(answer_pattern, gen_ans, re.M)
            if m:
                answer = m.group(1)
                return answer

        choices_dict = {}
        pattern = ""
        for c in "ABCD":
            choices_dict[str(data[f'{c}'])] = c
            pattern += re.escape(str(data[f'{c}']))+"|"
        pattern = pattern[:-1]
        m = re.findall(pattern, gen_ans, re.M)
        if len(m) >= 1:
            answer = choices_dict[m[0]]
            return answer

        return  None

    model,tokenizer,image_processor = load_model_with_processor(args.model_path)
    model.set_generation_config({"strategy":"greedy", "max_decode_steps":5})

    if model.is_chat_model == True:
        system_messages = []
        if model.system_message is not None:
            system_messages = [{"role":"system", "content":model.system_message}]

    device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
    model = model.to(device)
    
    outjson = {}
    correct,total = {}, {}
    for f in os.listdir(args.exam_data_path):
        key = re.sub("_(val|test)$", "", os.path.basename(os.path.splitext(f)[0])) 
        record = {}
        outjson[key] = record
         
        for data in read_data_shards(os.path.join(args.exam_data_path, f)):
            data_id = get_data_from_dict(data, ["id", "seq_id"], mode="first")
            question = get_data_from_dict(data, ["Question", "question"], mode="first")        
            
            if not question:
                continue

            answer = get_data_from_dict(data, ["Answer", "answer", "label"], mode="first")
            
            options = get_data_from_dict(data, ["options"], mode="first")
            option_a,option_b,option_c,option_d = None,None,None,None
            if options:
                if isinstance(options, list) and len(options) > 3:
                    option_a,option_b,option_c,option_d = options[:4]
                elif isinstance(options, dict):
                    option_a = get_data_from_dict(data, ["options,A"], mode="first")
                    option_b = get_data_from_dict(data, ["options,B"], mode="first")
                    option_c = get_data_from_dict(data, ["options,C"], mode="first")
                    option_d = get_data_from_dict(data, ["options,D"], mode="first")
            else:
                option_a = get_data_from_dict(data, ["A"], mode="first")
                option_b = get_data_from_dict(data, ["B"], mode="first")
                option_c = get_data_from_dict(data, ["C"], mode="first")
                option_d = get_data_from_dict(data, ["D"], mode="first")
 
            if not option_a or not option_b or not option_c or not option_d:
                continue
         
            prompt = "以下是中国关于学科考试的单项选择题，请选出其中的正确答案。"
            prompt += f"\n{question}"
            prompt += f"\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}"
            inputs = [prompt]

            if model.is_chat_model == True:
                messages = system_messages + [{"role":"user", "content":prompt}]
                inputs = [messages]
            
            for search_states in stream_generate(model=model,
                                                 inputs=inputs,
                                                 tokenizer=tokenizer,
                                                 device=device):
                pass

            gen_ans = search_states["text_buffer"][0]
            model_ans = extract_answer(
                    {"A":option_a, "B":option_b, "C":option_c, "D":option_d}, gen_ans)
            
            record[data_id] = model_ans

            if answer:
                total[key] = total.get(key, 0) + 1
                if model_ans and model_ans.lower() == answer.lower():
                    correct[key] = correct.get(key, 0) + 1

        if key in total:
            acc = correct[key] / total[key]
            logger.info(f"{key} done, total: {total[key]}, correct: {correct[key]}, acc: {acc:.4f}")

    total = sum(total[k] for k in total)
    correct = sum(correct[k] for k in correct)
    if total > 0:
        acc = correct / total
        logger.info(f"all done, total: {total}, correct: {correct}, acc: {acc:.4f}")
    
    with open(args.exam_output_path, "w", encoding="utf-8") as fout:
        fout.write(json.dumps(outjson, ensure_ascii=False, indent=4))


def main():
    logo_str = get_logo(concat=True)
    description=(
            f"\n{logo_str}\n"
    )
    usage = (
    '\nTo perform simple inference via the command line, run "python eval_exam.py '
    '--model_path <model_path> --exam_data_path <exam_data_path> --exam_output_path <exam_output_path>".\n'
    )
    parser = argparse.ArgumentParser(
            usage=usage,
            description=description,
            formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--model_path",
                        type=str,
                        help="")

    parser.add_argument('--exam_data_path',
                        type=str,
                        help="")

    parser.add_argument("--exam_output_path",
                        type=str,
                        help="")

    args = parser.parse_args(sys.argv[1:])

    eval_exam(args)


if __name__ == "__main__":

    main()
