import gradio as gr
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# MODEL = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
# MODEL = "beomi/KoAlpaca-Polyglot-5.8B"
# MODEL = "kyujinpy/Ko-PlatYi-6B"
# MODEL = "beomi/gemma-ko-7b"
# MODEL = "davidkim205/komt-mistral-7b-v1"
# MODEL = "beomi/Llama-3-Open-Ko-8B"
# MODEL = "upstage/SOLAR-10.7B-Instruct-v1.0"
# MODEL = "kakaobrain/kogpt"
# MODEL = "maum-ai/Llama-3-MAAL-8B-Instruct-v0.1"
MODEL = "/home/privacy/KoAlpaca/train_v1.1b/llama3-privacy-merged"
# MODEL = "/mnt/sda/privacy_backup/eeve-10.8b-privacy-sentence-dedupname"
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
    messages = state + [{"role": "user", "content": text}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    ans = pipe(
        inputs,
        do_sample=True,
        max_new_tokens=512,
        temperature=0.0001,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=terminators,
        repetition_penalty=1.2,
    )

    msg = ans[0]["generated_text"]
    
    state_chatbot = state_chatbot + [(text, msg)]

    print(state)
    print(state_chatbot)

    return state, state_chatbot, state_chatbot


with gr.Blocks(css="#chatbot .overflow-y-auto{height:750px}") as demo:
    state = gr.State([])
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

demo.queue().launch(debug=False, server_name="0.0.0.0", share=True)
