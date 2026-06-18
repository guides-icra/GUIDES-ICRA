import base64
import requests
import cv2
import numpy as np
import openai
import json
from robomimic.state_infuse.configs.prompts import prompt_configs

try:
    api_key = open('robomimic/state_infuse/configs/openai.key', 'r').read().replace('\n', "")
except FileNotFoundError as e:
    api_key = open('configs/openai.key', 'r').read().replace('\n', "")


openai.api_key = api_key

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def change_ndarray_to_base64(image, file_name='temp.png', write_image=False, with_image_format_change=True):
    # Convert the image to base64
    if with_image_format_change:
        image = (np.transpose(image, (1, 2, 0)) * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    if write_image:
        cv2.imwrite(file_name, cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR))

    return img_str


def get_internal_state_form_openai(image_left, image_hand, image_right, complete_rate, task,
                                   with_complete_rate=True, write_image=False, with_image_format_change=True):
    base64_image_left = change_ndarray_to_base64(image_left, task + f'_left_{complete_rate}.png',
                                                 write_image, with_image_format_change=with_image_format_change)
    base64_image_hand = change_ndarray_to_base64(image_hand, task + f'_hand_{complete_rate}.png',
                                                 write_image, with_image_format_change=with_image_format_change)
    base64_image_right = change_ndarray_to_base64(image_right, task + f'_right_{complete_rate}.png',
                                                  write_image, with_image_format_change=with_image_format_change)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_configs['pressMicronwaveButton'].format(**{'task': task, 'complete_rate': 100 * complete_rate})
                                if with_complete_rate else
                                prompt_configs['without_complete_rate'].format(**{'task': task})
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_left}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_hand}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_right}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100,
        "temperature": 0.1,
        "top_p": 1.0,
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print('error in get openai state analysis', e)
        return None


# Function to get embeddings for multiple strings
def get_embeddings(strings):
    model = "text-embedding-ada-002"
    embeddings = []
    url = "https://api.openai.com/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for string in strings:
        print(f"Getting embedding for: {string}")
        data = {
            "model": model,
            "input": string
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            embedding = response.json()['data'][0]['embedding']
            embeddings.append(np.array(embedding))
        else:
            print(f"Error: {response.status_code} - {response.text}")
            embeddings.append(None)
    return embeddings


if __name__ == '__main__':
    # Call the function and get embeddings
    # List of strings you want to get embeddings for
    strings = [
        "The quick brown fox jumps over the lazy dog.",
    ]

    # Choose the embedding model, e.g., "text-embedding-ada-002"

    embeddings = get_embeddings(strings)

    # Print the embedding for each string
    for i, embedding in enumerate(embeddings):
        print(f"Embedding for string {i + 1}: {embedding[:5]}... (truncated for brevity)")