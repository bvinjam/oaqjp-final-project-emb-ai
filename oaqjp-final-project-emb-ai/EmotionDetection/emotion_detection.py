"""Emotion detection helpers for the Flask app."""

import json

import requests

EMOTION_KEYWORDS = {
    "anger": ["angry", "mad", "rage", "furious", "annoyed", "hate"],
    "disgust": ["disgusted", "disgust", "gross", "repulsed", "revolted", "nasty"],
    "fear": ["afraid", "fear", "scared", "terrified", "panic", "worried"],
    "joy": ["glad", "happy", "joy", "love", "excited", "delighted", "great"],
    "sadness": ["sad", "cry", "unhappy", "grief", "lonely", "upset"],
}


def _empty_result():
    """Return a neutral result for invalid or empty inputs."""
    return {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }


def _keyword_based_emotion_detection(text_to_analyze):
    """Estimate an emotion using keyword matching if the external API fails."""
    normalized_text = text_to_analyze.lower()
    emotion_scores = {emotion: 0.0 for emotion in EMOTION_KEYWORDS}

    for emotion, keywords in EMOTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in normalized_text:
                emotion_scores[emotion] = 1.0
                break

    if not any(emotion_scores.values()):
        return {**emotion_scores, "dominant_emotion": None}

    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    return {**emotion_scores, "dominant_emotion": dominant_emotion}


def emotion_detector(text_to_analyze):
    """Detect the strongest emotion in a text string."""
    if not text_to_analyze or not text_to_analyze.strip():
        return _empty_result()

    url = (
        "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/"
        "NlpService/EmotionPredict"
    )
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    input_json = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(url, json=input_json, headers=headers, timeout=10)

        if response.status_code != 200:
            return _keyword_based_emotion_detection(text_to_analyze)

        response_dict = response.json()
        emotions = response_dict["emotionPredictions"][0]["emotion"]

        emotion_scores = {
            "anger": emotions["anger"],
            "disgust": emotions["disgust"],
            "fear": emotions["fear"],
            "joy": emotions["joy"],
            "sadness": emotions["sadness"],
        }

        dominant_emotion = max(emotion_scores, key=emotion_scores.get)

        return {
            "anger": emotion_scores["anger"],
            "disgust": emotion_scores["disgust"],
            "fear": emotion_scores["fear"],
            "joy": emotion_scores["joy"],
            "sadness": emotion_scores["sadness"],
            "dominant_emotion": dominant_emotion,
        }

    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError, TypeError):
        return _keyword_based_emotion_detection(text_to_analyze)
