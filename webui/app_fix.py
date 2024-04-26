import gradio as gr
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# MODEL = "EleutherAI/polyglot-ko-5.8b"
# MODEL = "beomi/Llama-3-Open-Ko-8B-preview"
# MODEL = "beomi/llama-2-ko-7b"
# MODEL = "TeamUNIVA/Komodo_7B_v1.0.0"
# MODEL = "beomi/Llama-3-Open-Ko-8B-Instruct-preview"
# MODEL = "chihoonlee10/T3Q-ko-solar-dpo-v6.0"
MODEL = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
# MODEL = "./train_v1.1b/eeve-10.8b-privacy-sentence"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype="auto",
    device_map="auto",
    # load_in_8bit=True,
    # revision="8bit",
    # max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # device=0,
)


def answer(state, state_chatbot, text):
    # messages = state + [{"role": "질문", "content": text}]
    messages = state + [{"role": "user", "content": text}]
    conversation_history = "\n".join(
        [f"### {msg['role']}:\n{msg['content']}" for msg in messages]
    )
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        # return_tensors="pt",
    )#.to(model.device)
    print(inputs)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    ans = pipe(
        inputs,
        do_sample=True,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=terminators,
    )

    msg = ans[0]["generated_text"]

    if "###" in msg:
        msg = msg.split("###")[0]

    new_state = [{"role": "이전 질문", "content": text}, {"role": "이전 답변", "content": msg}]

    state = state + new_state
    state_chatbot = state_chatbot + [(text, msg)]

    print(state)
    print(state_chatbot)

    return state, state_chatbot, state_chatbot


with gr.Blocks(css="#chatbot .overflow-y-auto{height:750px}") as demo:
    state = gr.State(
        [
            {
                "role": "system", 
                "content": "친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘."
            },
        ]
    )
    state_chatbot = gr.State([])

    with gr.Row():
        gr.HTML(
            """<div style="text-align: center; max-width: 500px; margin: 0 auto;">
            <div>
                <h1>ChatKoAlpaca 12.8B (v1.1b-chat-8bit)</h1>
            </div>
            <div>
                Demo runs on RTX 3090 (24GB) with 8bit quantized
            </div>
        </div>"""
        )

    with gr.Row():
        chatbot = gr.Chatbot(elem_id="chatbot")

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Send a message...", container=False)

    txt.submit(answer, [state, state_chatbot, txt], [state, state_chatbot, chatbot])
    txt.submit(lambda: "", None, txt)

demo.queue().launch(debug=True, server_name="0.0.0.0", share=True)
