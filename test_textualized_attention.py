# ===== File: test_textualized_attention.py =====
import os
import json
import base64
import io
import random

import openai
from dotenv import load_dotenv
import torch
from PIL import Image
import string

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import spacy
# load spaCy English model for POS tagging
nlp = spacy.load("en_core_web_sm")

# common words to ignore in attention tokens
STOPWORDS = {"question", "what", "is", "the", "a", "of", "and", "in", "to", "for", "it", "this", "that"}

from aokvqa_dataset import AOKVQADataset
from utils import load_config
from models.cross_attention_model import CrossAttentionExtractor

# 1) Setup
load_dotenv()
config = load_config("config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# help CUDA avoid fragmentation
torch.backends.cuda.max_split_size_mb = 8

# 2) Instantiate extractor on CPU & load checkpoint
extractor = CrossAttentionExtractor(
    vision_model_name=config["vision_model"],
    text_model_name=config["text_model"],
    num_labels=config.get("num_labels", 4),
)
chkpt = config.get("checkpoint_path", "checkpoint_supervised_epoch25.pt")
state = torch.load(chkpt, map_location="cpu")
extractor.load_state_dict(state)
extractor.eval()

# 3) Move text encoder and classifier to GPU in full precision
extractor.text_encoder.to(device)
if hasattr(extractor, "classifier"):
    extractor.classifier.to(device)

# 4) OpenAI client
client = openai.OpenAI(
    api_key=os.getenv("LITELLM_API_KEY"),
    base_url=config.get("litellm_base_url", "https://cmu.litellm.ai"),
)

def encode_image_b64(img: Image.Image, size=(128,128), quality=30) -> str:
    buf = io.BytesIO()
    tmp = img.copy()
    tmp.thumbnail(size)
    tmp.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def compute_attention_desc(question: str, image: Image.Image) -> str:
    # downsample for vision cross-attn
    img_small = image.copy()
    img_small.thumbnail((64,64))
    with torch.no_grad():
        _, attns = extractor([img_small], [question])
    # last layer [1, heads, seq_len, q_len]
    attn = attns[-1][0]             # [heads, seq_len, q_len]
    scores = attn.mean(dim=2).mean(dim=0)   # [seq_len]
    # tokenize question
    txt = f"Question: {question}"
    inputs = extractor.tokenizer(
        [txt],
        return_tensors="pt", padding=True, truncation=True, max_length=32
    ).to(device)
    tokens = extractor.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    # Filter out special and punctuation-only tokens
    special_tokens = set(extractor.tokenizer.all_special_tokens)
    token_scores = []
    for i, token in enumerate(tokens):
        # if token.startswith("##"):
        #     continue
        tok_lower = token.lower()
        # skip stopwords, special tokens, and punctuation-only tokens
        if tok_lower in STOPWORDS:
            continue
        if token in special_tokens:
            continue
        if all(ch in string.punctuation for ch in token):
            continue
        token_scores.append((token, scores[i].item()))
    # further filter to keep only noun tokens from the question
    doc = nlp(question)
    noun_set = {tok.text.lower() for tok in doc if tok.pos_ in {"NOUN", "PROPN"}}
    # keep only tokens whose lowercase form is in noun_set
    token_scores_nouns = [(token, score) for token, score in token_scores
                          if token.lower() in noun_set]
    if token_scores_nouns:
        token_scores = token_scores_nouns
    # Sort by attention score desc
    token_scores.sort(key=lambda x: x[1], reverse=True)
    # Select top-3 tokens
    top_tokens = [t for t, _ in token_scores[:5]]
    return "Top attended tokens (high to low) to think about potential rationales: " + ", ".join(top_tokens)

def test_vanilla_gpt4o_mc(question, mc_choices, image_path):
    """
    Returns (answer, attention_desc, rationales)
    """
    # load and thumbnail image
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((128,128))

    # compute attention description
    attention_desc = compute_attention_desc(question, img)

    # generate chain-of-thought rationales based on attention keywords
    # extract just the token list from the description
    keywords = attention_desc.split(":", 1)[1].strip()
    rationale_prompt = (
        f"Based on these keywords: {keywords}, "
        f"provide 2-3 concise rationale sentences that explain how you would answer the question: \"{question}\""
    )
    rationale_resp = client.chat.completions.create(
        model=config.get("gpt_model", "gpt-4o"),
        messages=[
            {"role": "system", "content": "You are a reasoning assistant. Provide step-by-step rationale."},
            {"role": "user", "content": rationale_prompt}
        ],
        temperature=0.7,
        max_tokens=64,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n\n"],
        n=1,
    )
    rationales = rationale_resp.choices[0].message.content.strip()

    # prepare image for prompt
    img_b64 = encode_image_b64(img)

    # build MC prompt
    full_prompt = (
        f"Question: {question}\n"
        f"{attention_desc}\n"
        f"Potential Rationales: {rationales}\n"
        f"A. {mc_choices[0]}\n"
        f"B. {mc_choices[1]}\n"
        f"C. {mc_choices[2]}\n"
        f"D. {mc_choices[3]}\n"
        f"Image:\n![image](data:image/jpeg;base64,{img_b64})\n\n"
        "Please respond with the single best answer text (not the letter)."
    )
    resp = client.chat.completions.create(
        model=config.get("gpt_model", "gpt-4o"),
        messages=[
            {"role":"system","content":"You are a visual question answering assistant. Choose the correct answer from the choices."},
            {"role":"user","content":full_prompt},
        ],
        temperature=0,
        top_p=0.9,
        max_tokens=128,
        frequency_penalty=0.2,
        presence_penalty=0.1,
        stop=["\n"],
        n=3,
    )
    answer = resp.choices[0].message.content.strip()

    # free small caches
    torch.cuda.empty_cache()
    return answer, attention_desc, rationales

def test_vanilla_gpt4o_da(question, image_path):
    """
    Returns (da_answer, rationales)
    """
    config = load_config("config.yaml")
    client = openai.OpenAI(
        api_key=os.getenv("LITELLM_API_KEY"),
        base_url=config.get("litellm_base_url", "https://cmu.litellm.ai"),
    )

    img = Image.open(image_path).convert("RGB")
    img.thumbnail((64, 64))

    # compute attention keywords
    attention_desc = compute_attention_desc(question, img)
    # generate concise rationales from attention keywords
    keywords = attention_desc.split(":", 1)[1].strip()
    rationale_prompt = (
        f"Based on these keywords: {keywords}, "
        f"provide 2 concise reasoning sentences to answer: \"{question}\""
    )
    rationale_resp = client.chat.completions.create(
        model=config.get("gpt_model", "gpt-4o"),
        messages=[
            {"role": "system", "content": "You are a reasoning assistant. Provide concise rationale."},
            {"role": "user", "content": rationale_prompt}
        ],
        temperature=0.7,
        max_tokens=64,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n\n"],
        n=1,
    )
    rationales = rationale_resp.choices[0].message.content.strip()

    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=30)
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('ascii')

    full_prompt = (
        f"Question: {question}\n"
        f"{attention_desc}\n"
        f"Potential Rationales: {rationales}\n"
        f"Image:\n![image](data:image/jpeg;base64,{img_b64})\n\n"
        "Please give the direct answer to the question based on the image. Make it as short as possible."
    )
    response = client.chat.completions.create(
        model=config.get("gpt_model", "gpt-4o"),
        messages=[
            {"role": "system", "content": "You are a visual question answering assistant. Given an image and a question, give the direct answer. Make the answer as short as possible."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0,
        top_p=0.9,
        max_tokens=64,
        frequency_penalty=0.2,
        presence_penalty=0.1,
        stop=["\n"],
        n=3,
    )
    da_ans = response.choices[0].message.content.strip()
    return da_ans, rationales

if __name__ == "__main__":
    # load dataset
    val_dataset = AOKVQADataset(
        aokvqa_dir=config["aokvqa_dir"],
        coco_dir=config["coco_dir"],
        split="val",
    )
    results = []

    for i in range(len(val_dataset)):
        print(f"Sample {i + 1}:")
        sample = val_dataset[i]

        question = sample['question']
        image_path = sample['image_path']
        mc_choices = [
            sample['mc_choices_0'],
            sample['mc_choices_1'],
            sample['mc_choices_2'],
            sample['mc_choices_3']
        ]
        mc_answers = sample['mc_answers']
        da_answers = sample['da_answers']
        # Ensure da_answers is a Python list (should be already, but for safety)
        if not isinstance(da_answers, list):
            da_answers = [da_answers]
        rationales = sample['rationales']
        qid = sample.get('question_id', i)

        # get MC and DA answers
        gpt_mc_answer, attention_desc, mc_rationales = test_vanilla_gpt4o_mc(question, mc_choices, image_path)
        gpt_da_answer, da_rationales = test_vanilla_gpt4o_da(question, image_path)

        # Evaluate DA correctness via GPT-4o
        refs = da_answers
        refs_str = ", ".join(refs)
        eval_prompt = (
            f"Direct answer: {gpt_da_answer}\n"
            f"Reference answers: {refs_str}\n"
            "Is the direct answer correct compared to the references? Respond with only 'correct' or 'incorrect'."
        )
        eval_resp = client.chat.completions.create(
            model=config.get("gpt_model", "gpt-4o"),
            messages=[
                {"role": "system", "content": "You are an evaluator assistant. Determine if the direct answer matches the reference answers."},
                {"role": "user", "content": eval_prompt}
            ],
            temperature=0,
        )
        da_correctness = eval_resp.choices[0].message.content.strip().lower()

        # record result
        results.append({
            'question_id': qid,
            'question': question,
            'image_path': image_path,
            'mc_choices': mc_choices,
            'correct_mc_answer': mc_answers,
            'mc_answer': gpt_mc_answer,
            'correct_da_answer': da_answers,
            'da_answer': gpt_da_answer,
            'attention_desc': attention_desc,
            'da_correctness': da_correctness,
            'mc_rationales': mc_rationales,
            'da_rationales': da_rationales,
        })

        # print summary
        print("Question:", question)
        print("Image Path:", image_path)
        print(f"Rationales: {rationales}")
        print(f"Choices: {', '.join(mc_choices)}")
        print(f"Correct MC Answer: {mc_answers} | GPT-4o MC Answer: {gpt_mc_answer}")
        print(f"Correct DA Answer: {da_answers} | GPT-4o DA Answer: {gpt_da_answer}")
        print(f"MC Rationales: {mc_rationales}")
        print(f"DA Rationales: {da_rationales}")
        print(f"DA correctness: {da_correctness}")
        print()

    out_json = config.get("output_json", "cross_attention_cot_results_with_correctness.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_json}")