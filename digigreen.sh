#!/bin/bash
# Classifier 
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "classifier",
      "messages": [
        {
            "role": "system",
            "content": "You are an agriculture expert.  Your job is to carefully read a question and an answer, and classify if the answer completely answers the question.\n\nIf the answer is relevant and complete to the provided question, output 1, otherwise output 0.  DO NOT output any explanation or prefix or suffix to the score.\n"
        },
        {
            "role": "user",
            "content": "Q: Is soil testing important for potato planting\nA: Yes, soil testing is important for potato planting. It helps farmers determine the nutrient levels in the soil and identify any deficiencies or imbalances. This information is crucial for determining the type and amount of fertilizers needed for the potato plants. Soil testing also helps identify the presence of pests, diseases, and other factors that may affect potato growth."
        }
    ],

    "max_tokens": 1024,
    "temperature": 0,
    "seed": 42,
    "response_format": {
      "type": "text",
      "value": "^[01]$"
    }
  }'



echo -e "\n\n"




# Relevance

curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "relevance",
 "messages": [
        {
            "role": "system",
            "content": "You are an agriculture expert.  Your job is to carefully read a question and fact from an answer to the question,\nand decide if the fact is relevant to the question or not.\n\nIf the fact is relevant to the provided question, output 1, otherwise output 0.  DO NOT output any explanation or prefix to the score.\n"
        },
        {
            "role": "user",
            "content": "Q: Is soil testing important for potato planting\nFact: Soil testing helps farmers identify nutrients"
        }
    ],

    "max_tokens": 1024,
    "temperature": 0,
    "seed": 42,
    "response_format": {
      "type": "text",
      "value": "^[01]$"
    }
  }'



echo -e "\n\n"

# extractor

curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
    "model": "extractor",
    "messages": [
        {
            "role": "system",
            "content": "You are an agriculture expert. Your job is to extract all agricultural facts from a user text; ignore the follow-up questions at the end, if any, and ignore all non-agricultural information.\nEach fact must be self-contained without pronouns or other contextual references.\nDO NOT INCLUDE any extra information not present in the message.\n\n{\"type\": \"object\", \"properties\": {\"facts\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"facts\"], \"additionalProperties\": false}\n"
        },
        {
            "role": "user",
            "content": "Soil testing is important for potato planting. It helps farmers determine the nutrient levels in the soil and identify any deficiencies or imbalances."
        }
    ],
    "max_tokens": 1024,
    "temperature": 0,
    "seed": 42,
    "response_format": {
        "type": "json_object",
        "value": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["facts"],
            "additionalProperties": false
        }
    }
}


EOF



