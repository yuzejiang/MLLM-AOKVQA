import json

def compute_accuracies(results):
    """
    Compute MC accuracy and DA accuracy.
    results: list of dicts with keys 'mc_answer', 'correct_mc_answer', 'da_answer', 'correct_da_answer'
    For DA, correct_da_answer may be a list of reference strings.
    """
    num = len(results)
    mc_correct = 0
    da_correct = 0

    for r in results:
        # MC: case-insensitive match
        pred_mc = r.get("mc_answer", "").strip().lower()
        gold_mc = str(r.get("correct_mc_answer", "")).strip().lower()
        if pred_mc == gold_mc:
            mc_correct += 1

        # DA: use da_correctness flag if available, otherwise fallback to text match
        da_flag = r.get("da_correctness")
        if isinstance(da_flag, str):
            if da_flag.strip().lower() == "correct":
                da_correct += 1

    mc_acc = mc_correct / num * 100 if num > 0 else 0.0
    da_acc = da_correct / num * 100 if num > 0 else 0.0
    return mc_correct, mc_acc, da_correct, da_acc

if __name__ == "__main__":
    # List of JSON result files to evaluate
    json_files = [
        "cross_attention_cot_results_with_correctness.json",
        "cross_attention_results_with_correctness.json",
        "gpt4o_results_with_correctness.json"
    ]

    for path in json_files:
        try:
            with open(path, "r") as f:
                results = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {path}")
            continue

        mc_corr, mc_acc, da_corr, da_acc = compute_accuracies(results)
        total = len(results)
        print(f"Results for {path}:")
        print(f"  Total samples: {total}")
        print(f"  MC Accuracy: {mc_corr}/{total} = {mc_acc:.2f}%")
        print(f"  DA Accuracy: {da_corr}/{total} = {da_acc:.2f}%")
        print()