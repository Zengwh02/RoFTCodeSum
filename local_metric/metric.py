from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import os
import smooth_bleu as bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def evaluate_bleu(reference, prediction, dir_):
    '''
    Evaluate BLEU score
    '''
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    p = []
    with open(os.path.join(dir_, "test_0.gold"), 'w', encoding='utf-8') as f:
        for idx, (predict, target) in enumerate(zip([prediction], [reference])):
            p.append(str(idx) + '\t' + predict)
            f.write(str(idx) + '\t' + target + '\n')

    (goldMap, predictionMap) = bleu.computeMaps(p, os.path.join(dir_, "test_0.gold"))
    test_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    return test_bleu

def calculate_bleu(predictions, references, dir_):
    '''
    Calculate BLEU score
    '''
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    p = []
    with open(os.path.join(dir_, "test_0.gold"), 'w', encoding='utf-8') as f:
        for idx, (predict, target) in enumerate(zip(predictions, references)):
            p.append(str(idx) + '\t' + predict)
            f.write(str(idx) + '\t' + target + '\n')

    (goldMap, predictionMap) = bleu.computeMaps(p, os.path.join(dir_, "test_0.gold"))
    test_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    return test_bleu


def evaluate_sbert(reference, prediction):
    '''
    Evaluate SBERT score
    '''
    embedding1 = model.encode(reference, convert_to_tensor=True)
    embedding2 = model.encode(prediction, convert_to_tensor=True)

    score = util.cos_sim(embedding1, embedding2).item()
    return round(score * 100, 4)

def calculate_sbert(references, predictions):
    '''
    Calculate SBERT score
    '''
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions, str):
        predictions = [predictions]

    embeddings1 = model.encode(references, convert_to_tensor=True)
    embeddings2 = model.encode(predictions, convert_to_tensor=True)

    scores = util.cos_sim(embeddings1, embeddings2)
    
    if len(references) > 1:
        scores = scores.diagonal()
    else:
        scores = scores[0][0]

    avg_score = scores.mean().item()
    return round(avg_score * 100, 2)


def calculate_meteor(predictions: list[str], references: list[str]) -> float:
    '''
    Calculate METEOR score
    '''
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = word_tokenize(pred)
        ref_tokens = word_tokenize(ref)
        score = meteor_score([ref_tokens], pred_tokens)
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    return round(avg_score * 100, 2)


def calculate_rouge_l(predictions: list[str], references: list[str]) -> float:
    '''
    Calculate ROUGE-L score
    '''
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, pred)['rougeL'].fmeasure for pred, ref in zip(predictions, references)]

    avg_score = sum(scores) / len(scores)
    return round(avg_score * 100, 2)


if __name__ == "__main__":
    pass
