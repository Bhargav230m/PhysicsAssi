import torch
import re

def manage_history(history: dict, current_count: int, add: bool=False, assistant_output: str=None, user_input: str=None) -> str:
    if add == True:
        history[f"message_{str(current_count)}"] = {
            "user_input": user_input,
            "assistant_output": assistant_output
        }
    
    history_lines = []
    
    for block_key in sorted(history.keys()):
        block = history[block_key]
        user = block["user_input"]
        assistant = block["assistant_output"]
        
        history_lines.append(f"<USER_INPUT>:{user}\n<ASSISTANT_OUTPUT>:{assistant}\n")
        
    return ''.join(history_lines)


def generate_response(llm, llm_tokenizer, stt_engine, tts_engine, device, saved_audio, history, current_history_count, assistant_count):
    message_template = """
    SYSTEM_MESSAGE:
    Your name is PhysicsAssi
    <ASSISTANT_OUTPUT> is PhysicsAssi and not the user
    <USER_INPUT> is user, After `<USER_INPUT>:` is user's input
    PhysicsAssi will MUST ONLY respond within <ASSISTANT_OUTPUT> tags
    PhysicsAssi will MUST NEVER generate <USER_INPUT> tags or content
    PhysicsAssi will MUST ONLY respond to the most recent <USER_INPUT>
    \n
    """

    result = stt_engine.transcribe(saved_audio)
    userInput = result["text"]
        
    history_s = manage_history(history, current_history_count)

    query = message_template + history_s + "<USER_INPUT>:" + userInput + "\n<ASSISTANT_OUTPUT>:\n"
    inputs = llm_tokenizer(query, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = llm.generate(**inputs, max_new_tokens=150, pad_token_id=llm_tokenizer.eos_token_id, do_sample=True, temperature=0.4, top_p=0.9)

    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    matches = re.findall(r'<ASSISTANT_OUTPUT>:\s*(.*?)(?=<ASSISTANT_OUTPUT>:|$)', response, re.DOTALL)

    if assistant_count <= len(matches):
        assistant_response = matches[assistant_count - 1].strip()
    else:
        assistant_response = "<SKIPPED>"

    manage_history(history, current_history_count, True, assistant_response, userInput)
    current_history_count += 1
    assistant_count += 1

    print(assistant_response)
    print("\n" + response)

    tts_engine.save_to_file(assistant_response, "response.wav")
    tts_engine.runAndWait()

    return assistant_response