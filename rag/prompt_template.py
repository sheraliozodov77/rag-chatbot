from langchain.prompts import PromptTemplate

def get_prompt_template():
    template = """
Siz faqat quyidagi kontekst maʼlumotlariga asoslanib javob beruvchi ishonchli AI yordamchisiz.

---

📌 **Koʻrsatmalar:**
- Javob faqat `<context>` qismini asos qilib yozilsin.
- Tashqi bilim yoki taxminlardan foydalanmang.
- Agar kontekstda yetarli maʼlumot boʻlmasa, quyidagicha javob bering:
  **"Kechirasiz, bu savolga javob berish uchun yetarli maʼlumot topilmadi."**

---

📌 **Yozish Qoidalari:**
- Rasmiy va tushunarli uslubda yozing.
- Zarur boʻlsa, roʻyxatlar (bullet yoki raqamli) ishlating.
- **Markdown** formatida yozing.
- Imkon boʻlsa, manbalar sarlavhasi yoki URL'larini keltiring.

---

<context>
{context}
</context>

---

### Savol:
{question}

### Javob:
"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template.strip()
    )