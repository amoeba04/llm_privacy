import gradio as gr
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# MODEL = "EleutherAI/polyglot-ko-5.8b"
# MODEL = "beomi/Llama-3-Open-Ko-8B-preview"
# MODEL = "beomi/llama-2-ko-7b"
# MODEL = "TeamUNIVA/Komodo_7B_v1.0.0"
# MODEL = "beomi/Llama-3-Open-Ko-8B-Instruct-preview"
# MODEL = "chihoonlee10/T3Q-ko-solar-dpo-v6.0"
# MODEL = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
# MODEL = "/home/privacy/KoAlpaca/train_v1.1b/eeve-10.8b-privacy-sentence"
# MODEL = "/home/privacy/KoAlpaca/train_v1.1b/eeve-10.8b-privacy-sentence-dedupname"
# MODEL = "/home/privacy/KoAlpaca/train_v1.1b/eeve-10.8b-privacy-sentence-dedupname-memorize"
MODEL = "/home/privacy/KoAlpaca/train_v1.1b/eeve-10.8b-privacy-phonenumber-dedupname-redup-level10"
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
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    ans = pipe(
        text,
        do_sample=True,
        max_new_tokens=512,
        temperature=0.0001,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=terminators,
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
