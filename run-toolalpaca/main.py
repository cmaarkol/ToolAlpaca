import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

llms = [
    "TangQiaoYu/ToolAlpaca-7B",
    # "TangQiaoYu/ToolAlpaca-13B"
    ]

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

print(args)

data_path = "run-toolalpaca/data.json"

with open(data_path, "r") as fp:
    data = json.load(fp)

for llm in llms:
    tokenizer = AutoTokenizer.from_pretrained(llm, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(llm, trust_remote_code=True).half()

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        device=args.device,
        do_sample=False
    )

    for service_id, service_data in enumerate(data):
        print(service_data["service_name"])
        for case_id, c in tqdm(enumerate(service_data["cases"])):
            instr = c["input"]
            output = generator(instr, return_full_text=False)[0]['generated_text']
            data[service_id]["cases"][case_id]["actual_output"] = output

    with open(f"run-toolalpaca/data_{llm.split('/')[-1]}.json", "w") as fp:
        json.dump([data], fp)