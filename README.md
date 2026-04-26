# LangChain ile Çoklu Zincir Kurma
> Birden fazla PromptTemplate'i LCEL ile sıralı zincire bağlama ve çıktıları birbirine taşıma — RAG serisinin 3. adımı

[![Colab'da Aç](https://img.shields.io/badge/Colab'da%20Aç-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/yasir237/rag-langchain-3/blob/main/rag_langchain_3.ipynb)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yasir-alrawi-12814521a/)

---

## Problem

Tek bir prompt çoğu zaman yetmez. Bir model çıktısını alıp başka bir modele farklı bir görevle göndermek istediğinde ne yapacaksın? Her zinciri elle mi çalıştıracaksın, çıktıyı elle mi kopyalayacaksın?

## Çözüm

LangChain'in `RunnableLambda` ve `RunnablePassthrough` yapıları, zincirler arasındaki veri akışını otomatikleştirir. Bir zincirin çıktısı, bir sonrakinin girdisine dönüşür — sen sadece akışı tanımlarsın. Bu yapı, RAG pipeline'larında **belge → işleme → yanıt** döngüsünün temelidir.

---

## Zincir Mimarisi

```
Girdi (concept: "autoencoder")
        │
        ▼
┌───────────────────────────────────────┐
│             chain_one                 │
│  PromptTemplate  →  Teknik açıklama   │
│  ChatGroq        →  Uzman dili        │
└───────────────────────────────────────┘
        │  AIMessage.content (ml_concept)
        ├─────────────────────┐
        ▼                     ▼
┌───────────────┐   ┌───────────────────────────────────────┐
│   chain_two   │   │             chain_three               │
│  Çocuk dili   │   │  PromptTemplate  →  Özetleme          │
│  Basit anlatım│   │  ChatGroq        →  Teknik TL;DR      │
└───────────────┘   └───────────────────────────────────────┘
        │                     │
        ▼                     ▼
   "anlatim"               "ozet"
        │                     │
        └──────────┬──────────┘
                   ▼
       {"anlatim": …, "ozet": …}
```

| Bileşen | Görevi |
|---|---|
| `RunnableLambda` | Zincir çıktısını bir sonraki zincirin beklediği formata çevirir |
| `RunnablePassthrough.assign` | Mevcut veriyi koruyarak yeni bir alan ekler |
| `chain_one → chain_two` | Teknik açıklama → çocuğa basit anlatım |
| `chain_one → chain_three` | Teknik açıklama → kısa teknik özet (TL;DR) |
| `overall_chain.invoke(...)` | Tüm akışı tek çağrıyla başlatır |

---

## Kullanılan Teknolojiler

![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=flat&logoColor=white)
![Llama](https://img.shields.io/badge/Llama_3.1_8B-0467DF?style=flat&logo=meta&logoColor=white)
![Python](https://img.shields.io/badge/Python_3-3776AB?style=flat&logo=python&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)

---

## Kurulum

```bash
pip install langchain langchain-core langchain-groq
```

### API Anahtarı

Google Colab **Secrets** sekmesine `GROQ_API_KEY` ekle.  
Groq API anahtarı almak için → [console.groq.com](https://console.groq.com)

---

## Kullanım

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq

chat = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

chain_one   = prompt_one | chat
chain_two   = prompt_two | chat
chain_three = prompt_three | chat

overall_chain = (
    chain_one
    | RunnableLambda(lambda x: {"ml_concept": x.content})
    | RunnablePassthrough.assign(
        child_explanation=lambda x: chain_two.invoke({"ml_concept": x["ml_concept"]}).content
    )
    | RunnableLambda(lambda x: {
        "st_concept": x["ml_concept"],          # chain_one çıktısı → teknik özet için
        "anlatim": x["child_explanation"]       # chain_two çıktısı → çocuğa anlatım
    })
    | RunnablePassthrough.assign(
        ozet=lambda x: chain_three.invoke({"st_concept": x["st_concept"]}).content
    )
)

response = overall_chain.invoke({"concept": "autoencoder"})
print(response["anlatim"])   # Çocuğa basit anlatım (chain_two)
print(response["ozet"])      # Teknik TL;DR        (chain_one → chain_three)
```

---

## Seri İçindeki Yeri

Bu notebook, LangChain ile kurulan RAG serisinin **3. adımıdır.**

```
[1] ✅ Mesaj yapısı ve LLM bağlantısı
[2] ✅ PromptTemplate ile şablonlu prompt
[3] ✅ Çoklu zincir kurma ve zincirleri bağlama   ← bu repo
[4]    Metin bölme ve embedding
[5]    Vektör store ve benzerlik araması
[6]    Function Calling                       
[7]    Uçtan uca RAG pipeline
```

Her adım bir sonrakine köprü kuruyor.  
Serinin tamamını takip etmek için LinkedIn profilimi ziyaret edebilirsin 👇

---

## Bağlantı

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yasir-alrawi-12814521a/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yasir237)

---

## Lisans

MIT
