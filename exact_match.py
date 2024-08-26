import pandas as pd
import torch 

# Read the CSV in chunks
# use_cols = ["TEXT"]

# chunk_size = 10000  # Adjust chunk size based on your memory
# for chunk in pd.read_csv('/home/hschung/Otter/docs/NOTEEVENTS.csv', chunksize=chunk_size):
#     print(chunk.head())  # Process each chunk


x = torch.load("/home/hschung/llama-multimodal-vqa/evaluation/mimic_m3ae_encoder_llama3_seed42_4gpus_0725_patching.pt")
print(x)

#post-processing assistant
def clean_assistant_responses(response):
    #remove 'assistant' prefix if present
    if response.lower().startswith('assistant'):
        response = response[len('assistant'):].strip()
    
    #handle assistant in the middle
    if response.lower() in ['no assistantno', 'yes assistantno']:
        return 'no'
    elif response.lower() in ['no assistantyes', 'yes assistantyes']:
        return 'yes'
    elif response.lower() in ['none assistantnone']:
        return 'none'
    
    #remove 'assistant' from middle of the rseponse
    response = response.replace('assistant', '')
    
    return response

#compare the scores between final_answers and ground_Truths 
def calculate_metrics(final_answers, ground_truth):
    """
    Calculate the exact match accuracy, precision, recall, and IoU between final answers and ground truth answers.
    Args:
        final_answers (list): List of model's predicted answers.
        ground_truth (list): List of ground truth answers.

    Returns:
        tuple: (exact_match_accuracy, precision, recall, iou)
            - exact_match_accuracy (float): Exact match accuracy as a percentage.
            - precision (float): Average precision across all examples.
            - recall (float): Average recall across all examples.
            - iou (float): Average Intersection over Union (IoU) across all examples.
    """
    assert len(final_answers) == len(ground_truth), "Length of final answers and ground truth must be the same."

    exact_matches = 0
    total_questions = len(final_answers)
    total_precision = 0
    total_recall = 0
    total_iou = 0

    for predicted, actual in zip(final_answers, ground_truth):
        predicted = predicted.replace("assistant", "").strip().lower()
        actual = actual.lower()
        
        predicted_elements = set(predicted.split(', '))
        actual_elements = set(actual.split(', '))
        matching_elements = predicted_elements.intersection(actual_elements)

        precision = len(matching_elements) / len(predicted_elements) if len(predicted_elements) > 0 else 0
        recall = len(matching_elements) / len(actual_elements) if len(actual_elements) > 0 else 0
        iou = len(matching_elements) / len(predicted_elements.union(actual_elements)) if len(predicted_elements.union(actual_elements)) > 0 else 0

        if all([precision == 1, recall == 1, iou == 1]):
            exact_matches += 1

        total_precision += precision
        total_recall += recall
        total_iou += iou

    accuracy = (exact_matches / total_questions) * 100
    average_precision = (total_precision / total_questions) * 100
    average_recall = (total_recall / total_questions) * 100
    average_iou = (total_iou / total_questions) * 100

    return accuracy, average_precision, average_recall, average_iou


# Clean all generated answers
# for category in x['generated_answers']:
#     x['generated_answers'][category] = [clean_assistant_responses(answer) for answer in x['generated_answers'][category]]

#Score calculation
for category in x['generated_answers']:
    accuracy, precision, recall, iou = calculate_metrics(x['generated_answers'][category], x['ground_truths'][category])
    print(f"Category: {category}")
    print(f"Exact Match Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"IoU: {iou:.2f}%")
    print()