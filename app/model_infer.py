import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
import re


class CNNEncoder(nn.Module):
    def __init__(self, name="densenet121", d_model=768):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=False, num_classes=0, global_pool="")
        c = self.backbone.feature_info[-1]["num_chs"]
        self.proj = nn.Conv2d(c, d_model, kernel_size=1)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.proj(feat)
        B, D, H, W = feat.shape
        return feat.flatten(2).transpose(1, 2).contiguous()


class VisionGPT2(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = GPT2Config.from_pretrained("gpt2")
        cfg.add_cross_attention = True
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", config=cfg)
        self.enc = CNNEncoder(d_model=cfg.n_embd)

    def clean_output(self, text):
        text = re.split(r'_{3,}', text)[0].strip()

        text = re.sub(r'\bof\s*_{2,}\s*,?', '', text)
        text = re.sub(r'\bDr\.\s*_{2,}\s*,?', 'Dr.', text)
        text = re.sub(r'\b_{2,}\s*', '', text)

        text = re.sub(r'\s{2,}', ' ', text)

        text = text.strip()
        if text and text[-1] not in '.!?':
            text += '.'

        text = re.split(r'NOTIFICATION:', text, flags=re.IGNORECASE)[0].strip()

        text = re.sub(r'\.\s*please\s+contact\.?$', '.', text, flags=re.IGNORECASE)

        return text

    def parse_findings_impression(self, text):
        findings = ""
        impression = ""

        text_lower = text.lower()

        imp_match = re.search(r'\bimpression\s*:', text, re.IGNORECASE)

        if imp_match:
            findings_part = text[:imp_match.start()].strip()
            impression_part = text[imp_match.end():].strip()

            findings = re.sub(r'^findings\s*:\s*', '', findings_part, flags=re.IGNORECASE).strip()
            impression = impression_part

            second_imp = re.search(r'\bimpression\s*:', impression, re.IGNORECASE)
            if second_imp:
                impression = impression[:second_imp.start()].strip()
        else:
            findings = re.sub(r'^findings\s*:\s*', '', text, flags=re.IGNORECASE).strip()

        return findings, impression

    @torch.inference_mode()
    def generate(self, image_tensor, tok, prompt="Findings: The", max_new_tokens=100,
                 top_p=0.9, top_k=50, temperature=0.70, repetition_penalty=1.20):
        self.eval()
        enc_states = self.enc(image_tensor)
        inp = tok(prompt, return_tensors="pt")
        input_ids = inp["input_ids"].to(image_tensor.device)
        attn_mask = inp["attention_mask"].to(image_tensor.device)

        out_ids = self.gpt2.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            encoder_hidden_states=enc_states,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

        raw_text = tok.decode(out_ids[0].tolist(), skip_special_tokens=True)
        cleaned_text = self.clean_output(raw_text)

        return cleaned_text


def build_transform(img_size=384):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(ckpt_path: str, device: str = "cpu"):
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    model = VisionGPT2()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    model.to(device)
    return model, tok


def preprocess_pil(pil_img: Image.Image, tfm):
    img = pil_img.convert("RGB")
    x = tfm(img).unsqueeze(0)
    return x