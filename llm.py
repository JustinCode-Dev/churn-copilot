import ollama
import joblib
import pandas as pd
import shap
import re

model = joblib.load('model/churn_model.pkl')
le_dict = joblib.load('data/le_dict.pkl')
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

y_proba = model.predict_proba(X_test)[:,1]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

def get_strength_label(value):
    if value < 0.1:
        return "slightly"
    elif value < 0.2:
        return "moderately"
    elif value < 0.45:
        return "strongly"
    else:
        return "very strongly"

def get_urgency(proba):
    if proba < 0.2:
        return "Low (monitor only, no action needed)"
    elif proba < 0.6:
        return "Moderate (watch closely, light-touch engagement optional)"
    else:
        return "High (recommend proactive retention action)"

def get_top_features(idx, X_test, shap_values, le_dict, top_n=5):
    shap_df = pd.DataFrame({
        'feature': X_test.columns,
        'feature_value': X_test.iloc[idx].values,
        'shap_value': shap_values[idx]
    })

    shap_df['abs_value'] = shap_df['shap_value'].abs()
    shap_df_sorted = shap_df.sort_values('abs_value', ascending=False).head(top_n)

    results = []
    for _, row in shap_df_sorted.iterrows():
        feature_name = row['feature']
        feature_value = row['feature_value']
        shap_value = row['shap_value']

        if feature_name in le_dict:
            readable_value = le_dict[feature_name].classes_[int(feature_value)]

        else:
            readable_value = feature_value

        strength = get_strength_label(shap_value)
        
        direction = f'{strength} pushes toward churn' if shap_value > 0 else f'{strength} pushes toward staying'

        results.append({
            'feature': feature_name,
            'value': readable_value,
            'direction': direction,
        })

    return results

def seperate_llm_output(output):
    match_reasoning = re.search(r"Reasoning:\s*(.*?)(?=\s*Recommendation:|$)", output, re.DOTALL | re.IGNORECASE)
    match_recommendation = re.search(r"Recommendation:\s*(.*)", output, re.DOTALL | re.IGNORECASE)

    reasoning = match_reasoning.group(1).strip() if match_reasoning else output
    recommendation = match_recommendation.group(1).strip() if match_recommendation else "-"
    
    return reasoning, recommendation

def build_user_prompt(proba, top_features):
    factors_text = ""
    urgency = get_urgency(proba)
    for f in top_features:
        factors_text += f"- {f['feature']}: {f['value']} ({f['direction']})\n"

    prompt = f"""Customer churn probability: {proba*100:.1f}%
Urgency level: {urgency}
Key contributing factors:
{factors_text}
Please provide your analysis."""
    
    print("-----"*30)
    print(prompt)
    print("-----"*30)
    return prompt

system_prompt = """You are a Customer Success assistant. Analyze the customer churn probability and the top 3 contributing factors provided.

Respond ONLY in this exact format, starting directly with "Reasoning:":

Reasoning: [1-2 concise sentences summarizing how the top 3 factors drive or reduce the churn risk.]
Recommendation: [1 short, specific action tailored to the risk level.]

Example Input:
Customer churn probability: 5.8%
Urgency level: Low (monitor only, no action needed)
Key contributing factors:
- Contract: Two year (slightly pushes toward staying)
- tenure: 72.0 (slightly pushes toward staying)
- MonthlyCharges: 114.05 (very strongly pushes toward churn)

Example Output:
Reasoning: The customer's long tenure and two-year contract strongly push toward staying, outweighing the moderate churn pressure from a high monthly charge of $114.05.
Recommendation: Monitor for price sensitivity, but no urgent action needed given strong retention signals.

Example Input:
Customer churn probability: 93.0%
Urgency level: High (recommend proactive retention action)
Key contributing factors:
- MonthlyCharges: 100.15 (very strongly pushes toward churn)
- Contract: Month-to-month (very strongly pushes toward churn)
- tenure: 8.0 (very strongly pushes toward churn)

Example Output:
Reasoning: All three key factors high monthly charges, a month-to-month contract, and short tenure push strongly toward churn, with no mitigating factors present.
Recommendation: Prioritize immediate outreach with a retention offer, such as a discounted long-term contract, before the customer churns.
"""

def analyze_customer(idx=10):
    proba = y_proba[idx]
    top_features = get_top_features(idx, X_test, shap_values, le_dict, top_n=3)
    user_prompt = build_user_prompt(y_proba[idx], top_features)

    response = ollama.chat(
        model= 'qwen3:1.7b', 
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        think=False,
        options={
            'temperature': 0.4
        }
    )

    result = response['message']['content']
    reasoning, recommendation = seperate_llm_output(result)

    return {
        "churn_prob": round(proba * 100),
        "reasoning": reasoning,
        "recommendation": recommendation
    }
def analyze_customer_behaviour(customer_message):
    behaviour_system_prompt = """You are an emotion classifier for Customer Service chats. Analyze the customer's message and classify their behaviour into ONE exact word from this list:

- Aggressive: Hostile, furious, using harsh words, or strongly demanding to cancel immediately.
- Upset: Frustrated, disappointed, or complaining about high charges/bad service.
- Skeptical: Doubtful, hesitant, questioning the offer, or worried about hidden terms.
- Neutral: Calm, matter-of-fact, asking general questions, or undecided.
- Receptive: Warming up, interested in the solution/discount, or willing to listen.
- Happy: Satisfied, grateful, relieved, or accepting the retention offer.

Respond with ONLY ONE WORD from the list above. Do not add punctuation or explanations.
Examples:
Customer: "This bill is ridiculous! Cancel my account right now, I'm done with this scam!"
Output: Aggressive

Customer: "My monthly charge went up again and I really can't afford this anymore."
Output: Upset

Customer: "A 20% discount sounds okay, but does that lock me into a long contract with hidden fees?"
Output: Skeptical

Customer: "How much would the new plan cost per month?"
Output: Neutral

Customer: "Oh, if there's no cancellation fee, I might actually consider staying."
Output: Receptive

Customer: "That works great for me! Thank you so much for sorting this out."
Output: Happy
"""

    response = ollama.chat(
        model='qwen3:1.7b',
        messages=[
            {'role': 'system', 'content': behaviour_system_prompt},
            {'role': 'user', 'content': customer_message}
        ],
        think=False,
        options={
            'temperature': 0.1,
            'num_predict': 10
        }
    )
    valid_labels = ["Aggressive", "Upset", "Skeptical", "Neutral", "Receptive", "Happy"]
    raw_response = response['message']['content'].strip().replace(".", "").capitalize()

    return raw_response if raw_response in valid_labels else "Neutral"

if __name__ == "__main__":
    print(analyze_customer(idx=10))