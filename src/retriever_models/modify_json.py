import json

input_path = "/scratch/as20410/NLP_project/self-biorag/retriever/output/medcpt_top10_evidence.json"
output_path = "/scratch/as20410/NLP_project/self-biorag/retriever/output/medcpt_top10_evidence_corrected.json"

with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
    f_out.write("[\n")
    first = True
    for line in f_in:
        if not first:
            f_out.write(",\n")
        f_out.write(line.strip())
        first = False
    f_out.write("\n]")