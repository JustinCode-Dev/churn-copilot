import ollama
import json
def chat_simulator(churn_proba, reasoning ,chat_history):
    sim_system_prompt = f"""You are roleplaying as a telecom customer chatting with a Customer Service (CS) Agent.

Your hidden customer profile:
- Churn Risk Probability: {churn_proba}%
- Your main frustration / situation: {reasoning}

Roleplay Rules:
1. Speak in casual, natural everyday English (1 to 2 short sentences max, like a real text message).
2. You are the CUSTOMER, NOT a support agent. NEVER say phrases like "Let me know if you need anything else" or "How can I help you?".
3. Never mention percentages, SHAP, or "reasoning". Act like a real human customer.
4. If the conversation hasn't started yet, send the first message complaining about or asking about your main issue based on your situation.
5. If the CS Agent replies to you:
   - If they ignore your issue or refuse to help, stay frustrated or threaten to leave.
   - If they offer a reasonable discount, perk, or solution that fixes your frustration, gradually soften up, ask clarifying questions, or happily accept the offer."""
    
    messages = [{'role': 'system', 'content': sim_system_prompt}]

    if not chat_history:
        messages.append({
            'role': 'user', 
            'content': "Start the chat by sending your first message to Customer Service."
        })
    else:
        # User = CS Agent, Assistant = Customer
        for msg in chat_history:
            messages.append({'role': msg['role'], 'content': msg['content']})

    response = ollama.chat(
        model= 'qwen3:1.7b', 
        messages=messages,
        think=False,
        options={
            'temperature': 0.6,
            'repeat_penalty': 1.3,
            'num_predict': 200
        }
    )
    return response['message']['content'].strip().strip('"')