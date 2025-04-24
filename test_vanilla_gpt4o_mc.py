import base64
import io
import os
import random
import json

import openai
from dotenv import load_dotenv
from PIL import Image
import torch

from aokvqa_dataset import AOKVQADataset
from utils import load_config

load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_vanilla_gpt4o_mc(question, mc_choices, image_path):
    config = load_config("config.yaml")
    client = openai.OpenAI(
        api_key=os.getenv("LITELLM_API_KEY"),
        base_url="https://cmu.litellm.ai",
    )

    img = Image.open(image_path).convert('RGB')
    img.thumbnail((64, 64))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=30)
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('ascii')

    full_prompt = (
        f"Question: {question}\n"
        f"A. {mc_choices[0]}\n"
        f"B. {mc_choices[1]}\n"
        f"C. {mc_choices[2]}\n"
        f"D. {mc_choices[3]}\n"
        f"Image:\n![image](data:image/jpeg;base64,{img_b64})\n\n"
        # f"Please give the direct answer to the question based on the image.\n"
        "Please respond with the single best answer text (not the letter). Only include the answer in your response."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a visual question answering assistant. Given an image and a multiple-choice question, select the correct answer."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0,
        top_p=0.9,
        max_tokens=128,
        frequency_penalty=0.2,
        presence_penalty=0.1,
        stop=["\n"],
        n=3,
    )
    answer = response.choices[0].message.content.strip()
    return answer


def test_vanilla_gpt4o_da(question, image_path):
    config = load_config("config.yaml")
    client = openai.OpenAI(
        api_key=os.getenv("LITELLM_API_KEY"),
        base_url="https://cmu.litellm.ai",
    )

    img = Image.open(image_path).convert('RGB')
    img.thumbnail((64, 64))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=30)
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('ascii')

    full_prompt = (
        f"Question: {question}\n"
        f"Image:\n![image](data:image/jpeg;base64,{img_b64})\n\n"
        f"Please give the direct answer to the question based on the image. Make the answer as short as possible.\n"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a visual question answering assistant. Given an image and a question, give the direct answer. Make the answer as short as possible."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0,
        top_p=0.9,
        max_tokens=128,
        frequency_penalty=0.2,
        presence_penalty=0.1,
        stop=["\n"],
        n=3,
    )
    answer = response.choices[0].message.content.strip()
    return answer


if __name__ == "__main__":
    config = load_config("config.yaml")
    client = openai.OpenAI(
        api_key=os.getenv("LITELLM_API_KEY"),
        base_url="https://cmu.litellm.ai",
    )
    results = []

    val_dataset = AOKVQADataset(
        aokvqa_dir="datasets/aokvqa",
        coco_dir="datasets/coco",
        split="val"
    )

    for i in range(len(val_dataset)):
        print(f"Sample {i + 1}:")
        # sample = random.choice(val_dataset)
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
        mc_answers_index = sample['mc_answers_index']
        da_answers = sample['da_answers']
        rationales = sample['rationales']

        gpt_mc_answer = test_vanilla_gpt4o_mc(question, mc_choices, image_path)
        get_da_answer = test_vanilla_gpt4o_da(question, image_path)
        # Evaluate DA correctness via GPT-4o
        refs = da_answers if isinstance(da_answers, list) else [da_answers]
        refs_str = ", ".join(refs)
        eval_prompt = (
            f"Direct answer: {get_da_answer}\n"
            f"Reference answers: {refs_str}\n"
            "Is the direct answer correct compared to the references? Respond with only 'correct' or 'incorrect'."
        )
        eval_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an evaluator assistant. Determine if the direct answer matches the reference answers."},
                {"role": "user", "content": eval_prompt}
            ],
            temperature=0.0,
        )
        da_correctness = eval_resp.choices[0].message.content.strip().lower()
        
        # record result
        qid = sample.get('question_id', i)
        results.append({'question_id': qid, 
                        'question': question,
                        'image_path': image_path,
                        'mc_choices': mc_choices,
                        'correct_mc_answer': mc_answers,
                        'mc_answer': gpt_mc_answer, 
                        'correct_da_answer': da_answers,
                        'da_answer': get_da_answer,
                        'da_correctness': da_correctness,
                        })
        
        print("Question:", question)
        print("Image Path:", image_path)
        print(f"Rationales: {rationales}")
        print(f"Choices: {mc_choices[0]}, {mc_choices[1]}, {mc_choices[2]}, {mc_choices[3]}")
        print(f"Correct MC Answer: {mc_answers} | GPT-4o Answer: {gpt_mc_answer}")
        print(f"Correct DA Answer: {da_answers} | GPT-4o DA Answer: {get_da_answer}")
        print(f"DA correctness: {da_correctness}")
        print()

    # save results to JSON
    output_path = config.get('output_json', 'gpt4o_results_with_correctness.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_path}")