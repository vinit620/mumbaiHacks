import json

import numpy as np
import torch
from huggingface_hub import login
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline

token = "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"
model_id = "meta-llama/Llama-3.2-3B-Instruct"

login(token=token, add_to_git_credential=True)

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


JSON_PATH = "MOCK_DATA_1.json"


def read_json_file(file_path):
    with open(file_path, "r") as file:
        records = json.load(file)
    return records


records = read_json_file(JSON_PATH)


# Define the parameters
parameters = [
    "cashflow",
    "sip",
    "gov_funds",
    "equity",
    "commodities",
    "bonds",
    "fixed_deposite",
    "insurance",
]

# Extract parameter values and handle missing values
values = []
for person in records:
    person_values = []
    for param in parameters:
        value = person.get(param, 0) if person.get(param) is not None else 0
        person_values.append(value)
    values.append(person_values)

# Convert to numpy array for easier manipulation
values = np.array(values)

# Normalize the values
scaler = MinMaxScaler()
normalized_values = scaler.fit_transform(values)


def generate_weights(person_data):
    messages = [
        {
            "role": "system",
            "content": "You are an AI model that generates weights for calculating a credibility score based on the following parameters: "
            "cashflow, sip, gov_funds, equity, commodities, bonds, fixed_deposite, and insurance. "
            "The weights should be higher for parameters that have non-null values and should sum up to 1. "
            "Consider the interdependence of these parameters. For example, if a person doesn't have insurance but has a high amount in fixed deposits or SIP, "
            "the score should still be high due to financial stability. !",
        },
        {
            "role": "user",
            "content": "Here is the data for a person:\n"
            f"{person_data}\n"
            "Generate a list of weights for these parameters. List of weights should be comma seperated in the same order as person data parameters, do not return any additional information other than comma seperated numbers.",
        },
    ]

    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    return outputs[0]["generated_text"][-1]["content"]


# Generate weights and calculate credibility score for 3 people
count = 0
for i, person in enumerate(records):
    person_data = {param: person.get(param, 0) for param in parameters}
    weights = generate_weights(person_data).split(",")

    # Normalize the weights so that they sum up to 1
    weights = [float(weight) for weight in weights]
    weights = [weight / sum(weights) for weight in weights]

    # Calculate the credibility score
    norm_vals = normalized_values[i]
    score = sum(norm_vals[j] * weights[j] for j in range(len(parameters)))

    person["credibility_score"] = score

    print(f"Person {i+1}: {person}")
    print(f"Credibility score for person {i+1}: {score}")
    count += 1
    if count == 3:
        break
