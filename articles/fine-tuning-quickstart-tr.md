# LLM Fine-Tuning'e Hızlı Başlangıç: LoRA, PEFT ve QLoRA'yı Anlamak (Google Colab Notebook'u ile)

*İlk Large Language Model'inizi ücretsiz bir GPU üzerinde fine-tune edin — ve perde arkasında gerçekte ne olduğunu da anlayın.*

---

## İçindekiler

1. [Bu yazı neden var?](#bu-yazı-neden-var)
2. [Fine-tuning nedir?](#fine-tuning-nedir)
3. [Sorun: full fine-tuning pahalıdır](#sorun-full-fine-tuning-pahalıdır)
4. [PEFT: Parameter-Efficient Fine-Tuning](#peft-parameter-efficient-fine-tuning)
5. [LoRA'yı temelden anlamak](#loranın-temelden-anlamak)
6. [QLoRA: 4-bit bütçeyle LoRA](#qlora-4-bit-bütçeyle-lora)
7. [Uçtan uca workflow](#uçtan-uca-workflow)
8. [Koda adım adım bakış](#koda-adım-adım-bakış)
9. [Hyperparameter seçimi](#hyperparameter-seçimi)
10. [Modelinizi değerlendirmek (evaluation)](#modelinizi-değerlendirmek-evaluation)
11. [Deployment: adapter mı, merged model mı?](#deployment-adapter-mı-merged-model-mı)
12. [Sık yapılan hatalar](#sık-yapılan-hatalar)
13. [Buradan sonrası](#buradan-sonrası)

---

## Bu yazı neden var?

Daha önce "işte verinle modeli fine-tune et" cümlesini okuyup *nasıl* yapılacağını merak ettiyseniz, bu yazı tam size göre. Sonunda şunları yapabilir hâle geleceksiniz:

- **Fine-tuning'in ne olduğunu** ve gerçekte ne zaman gerektiğini anlamak.
- **LoRA**, **PEFT** ve **QLoRA** kavramlarını — birer moda terim olarak değil, somut mekanizmalar olarak — anlamak.
- **Ücretsiz bir Google Colab GPU'su** üzerinde gerçek bir modeli fine-tune eden, **çalışan eksiksiz bir notebook'u** baştan sona koşturmuş olmak.

Yazıya eşlik eden notebook, [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) modelini [`yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned) instruction dataset'i üzerinde fine-tune eder. Her şey ücretsiz bir **T4 (16 GB)** GPU'ya sığar.

---

## Fine-tuning nedir?

Modern bir LLM birkaç aşamada eğitilir:

1. **Pre-training** — model, internetin devasa bir kesitini okur ve bir sonraki token'ı tahmin etmeyi öğrenir. Dil bilgisini, olguları, akıl yürütme örüntülerini ve dünya bilgisini bu aşamada kazanır. Milyonlarca dolara ve binlerce GPU'ya mal olur.
2. **Fine-tuning** — bu zaten yetenekli modeli alıp, istediğimiz gibi davranması için *daha küçük ve hedefe yönelik* bir dataset üzerinde eğitmeye devam ederiz.

**Fine-tuning aslında sadece ek bir training'dir; ama önemsediğiniz veri üzerinde yapılır.** Pre-trained bir modelin weight'leri başlangıç noktasıdır; gradient descent bunları sizin task'inize doğru nazikçe iter.

### Ne zaman fine-tune etmeli?

Fine-tuning, modelin **davranışını, formatını, tonunu veya bir becerisini** değiştirmek istediğinizde doğru araçtır — örneğin:

- Her zaman belirli bir JSON schema ile cevap vermek.
- Bir marka sesi veya alana özgü bir üslup (hukuk, tıp, destek) benimsemek.
- Prompt ile ifade etmesi zor bir task'i öğrenmek (örneğin niş bir classification şeması).
- Düşük kaynaklı (low-resource) bir dilde talimatları daha iyi takip etmek.

### Ne zaman fine-tune *etmemeli*?

Asıl ihtiyacınız modelin **yeni olguları bilmesi** ise (şirketinizin dokümanları, günün haberleri), genellikle **Retrieval-Augmented Generation (RAG)**, fine-tuning'den daha ucuz ve daha kolay sürdürülebilir bir çözümdür. İyi yazılmış bir prompt ya da birkaç few-shot örneği işi zaten görüyorsa oradan başlayın — prompting bedavadır.

> **Pratik kural:** Prompting *ne sorduğunuzu* değiştirir. RAG *modelin ne görebildiğini* değiştirir. Fine-tuning ise *modelin ne olduğunu* değiştirir.

---

## Sorun: full fine-tuning pahalıdır

"Full fine-tuning", modeldeki **her** weight'i güncellemek demektir. 16-bit precision'da 7B parametreli bir model için bu kabaca şu demektir:

- Sadece weight'leri tutmak için **~14 GB**.
- Gradient'ler için **~14 GB** (her weight başına bir tane).
- Adam optimizer state'i için **~56 GB** (her weight için fp32'de iki ek değer tutar).

Yani daha bir batch veri bile yüklemeden **80+ GB VRAM**. Bu yüzden 7B'lik bir modeli full fine-tune etmek A100/H100 sınıfı bir GPU ister. Çoğu birey ve küçük ekip için bu en baştan imkânsızdır.

**PEFT** ve **LoRA**, tam olarak bu sorunu çözmek için tasarlandı.

---

## PEFT: Parameter-Efficient Fine-Tuning

**PEFT (Parameter-Efficient Fine-Tuning)**, bir modeli yalnızca parametrelerinin **çok küçük bir kısmını** eğiterek, geri kalanı **frozen (donmuş)** tutarak fine-tune eden yöntemler ailesinin genel adıdır.

Temel içgörü şu: bir modeli özelleştirmek için tüm weight'leri oynatmanız gerekmez. Dev pre-trained omurgayı (**backbone**) **dondurup**, onu "yönlendiren" az sayıda yeni veya seçili parametreyi eğitebilirsiniz.

PEFT yöntemleri arasında LoRA, prefix tuning, prompt tuning, adapter'lar, IA³ ve daha fazlası vardır. Pratikte **açık ara en popüler olanı LoRA'dır** ve Hugging Face'in [`peft`](https://github.com/huggingface/peft) kütüphanesi bunu birkaç satır koda indirger.

Bu neden önemli?

- **Memory:** optimizer state'i yalnızca *trainable* parametreler için saklarsınız (çoğu zaman modelin <%1'i), böylece yukarıdaki 80 GB sorunu dramatik biçimde küçülür.
- **Storage ve paylaşım:** training'in çıktısı, koca bir GB'larca model kopyası yerine küçük bir **adapter**'dır (çoğu zaman birkaç MB).
- **Modülerlik:** tek bir frozen base model tutup, üzerine task'e özgü birçok adapter takıp çıkarabilirsiniz.

---

## LoRA'yı temelden anlamak

**LoRA**, **Low-Rank Adaptation** ifadesinin kısaltmasıdır. 2021 tarihli *"LoRA: Low-Rank Adaptation of Large Language Models"* makalesinden gelir.

### Temel gözlem

Bir modeli fine-tune ettiğinizde, her weight matrisi `W` bir miktar `ΔW` kadar değişir ve size yeni bir matris `W + ΔW` verir. LoRA'nın yazarları, bu güncellemenin (`ΔW`) **düşük intrinsic rank**'e sahip olduğunu gözlemledi — yani çok daha küçük iki matrisin çarpımı ile iyi bir şekilde yaklaşıklanabilir.

### Matematik (göründüğünden basit)

`d × k` boyutunda bir weight matrisi `W` alalım. LoRA, tüm `ΔW`'yi (ki `d × k` tane değeri vardır) öğrenmek yerine güncellemeyi şöyle temsil eder:

```
ΔW = B · A
```

burada:
- `A` matrisinin boyutu `r × k`
- `B` matrisinin boyutu `d × r`
- `r` (**rank**) küçüktür — tipik olarak 8, 16, 32 veya 64.

Forward pass sırasında katman şunu hesaplar:

```
h = W·x + (B·A)·x · (alpha / r)
```

- `W` **frozen** kalır (onun için asla gradient hesaplamayız).
- Yalnızca `A` ve `B` **trainable**'dır.
- `alpha`, adapter'ın çıktıyı ne kadar güçlü etkilediğini kontrol eden bir scaling factor'dür.

### Bu neden bu kadar çok tasarruf sağlar?

Diyelim ki `W` matrisi `4096 × 4096` ≈ **16,7 milyon** parametre. `r = 16` ile:
- `A` matrisi `16 × 4096` = 65.536 parametre
- `B` matrisi `4096 × 16` = 65.536 parametre
- Toplam: **131.072** trainable parametre — orijinalin yaklaşık **%0,8'i**.

Bu tasarrufu her katmana yayın; tipik olarak modelin **%1'inin çok altını** eğitiyor olursunuz. Training'in başında `B` sıfır olarak initialize edilir, dolayısıyla `B·A = 0` olur ve model *tam olarak* orijinali gibi davranır — training sonra oradan başlayarak adapter'ı nazikçe öğrenir.

### Sezgi

Frozen base model'i parlak bir genel uzman gibi düşünün. LoRA, belirli katmanların önüne küçük, trainable bir "lens" ekler. Uzmanın beynini baştan yazmıyorsunuz — ona odaklanmış yeni bir alışkanlık öğretiyorsunuz ve bu alışkanlığı saklamak ucuz, kaldırmak veya değiştirmek kolaydır.

### Hangi katmanlar adapter alır?

**Target modules**'ü siz seçersiniz. Transformer tabanlı LLM'lerde attention projeksiyonları (`q_proj`, `k_proj`, `v_proj`, `o_proj`) ve MLP projeksiyonları (`gate_proj`, `up_proj`, `down_proj`) olağan hedeflerdir. Daha fazla modülü adapt etmek, modele uyum sağlaması için daha fazla kapasite verir; karşılığında küçük bir ek maliyet gelir.

---

## QLoRA: 4-bit bütçeyle LoRA

LoRA, **trainable** parametreleri zaten küçültür. Ama **frozen** base model hâlâ memory'de durmak zorundadır — ve 16-bit'te 7B'lik bir model ~14 GB'dır. **QLoRA** ("Quantized LoRA", 2023) son adımı atar: frozen base weight'lerini **4-bit precision** ile saklar.

QLoRA üç fikri birleştirir:

1. **4-bit NF4 quantization** — sinir ağlarının normal dağılımlı weight'leri için ayarlanmış 4-bit'lik bir veri tipi ("NormalFloat"). Bu, frozen modelin memory'sini kabaca **dörtte birine** indirir.
2. **Double quantization** — quantization sabitleri bile quantize edilerek biraz daha memory kazanılır.
3. **Paged optimizers** — optimizer state'i, memory zirvelerini atlatmak için CPU RAM'e taşabilir.

Kritik nokta: **forward ve backward matematiği hâlâ 16-bit'te** (`bf16`/`fp16`) yapılır. 4-bit weight'ler, hesaplama için anlık olarak de-quantize edilir. Yani 4-bit storage'ın memory tasarrufunu, 16-bit compute'un sayısal kararlılığıyla birlikte elde edersiniz.

**Sonuç:** 7B'lik bir modeli tek bir 16 GB GPU üzerinde fine-tune edebilirsiniz — ki ücretsiz Colab katmanını gerçek fine-tuning için kullanılabilir kılan tam da budur.

> Kısacası: **PEFT** ailedir, **LoRA** yöntemdir, **QLoRA** ise quantize edilmiş (4-bit) frozen base'e sahip LoRA'dır.

---

## Uçtan uca workflow

Hangi framework kullanılırsa kullanılsın, her fine-tuning projesi aynı iskelete sahiptir:

1. **Bir base model seçin** — genellikle bir *instruct* varyantı, çünkü chat formatını zaten anlar.
2. **Bir dataset seçin / hazırlayın** — istediğiniz davranışı tanımlayan örnekler.
3. **Veriyi formatlayın** — ham satırları modelin **chat template**'ine dönüştürün.
4. **Modeli verimli yükleyin** — memory için 4-bit quantization (QLoRA).
5. **Adapter'ları takın** — LoRA'yı PEFT ile yapılandırın.
6. **Train edin** — makul hyperparameter'larla training loop'unu çalıştırın.
7. **Evaluate edin** — çıktıları öncesi/sonrası karşılaştırın; held-out prompt'larda test edin.
8. **Save / merge / deploy edin** — adapter'ı yayınlayın veya bağımsız bir modele merge edin.

Bu repodaki notebook bu adımların her birini uygular. Önemli kısımlara birlikte bakalım.

---

## Koda adım adım bakış

### 1. Base model'i 4-bit yükleyin

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
```

`BitsAndBytesConfig`, düz bir yüklemeyi **QLoRA** yüklemesine dönüştüren şeydir. Model artık 16-bit ayak izinin yalnızca bir kesrini kaplar.

### 2. Dataset'i chat template ile formatlayın

Instruct modeller, özel rol işaretleyicileriyle (system / user / assistant) eğitilir. Doğru fine-tune etmek için verinizin de **aynı** formatı kullanması gerekir. Tokenizer, modelin template'ini bilir:

```python
def to_chat_text(example):
    user = example["instruction"]
    if example.get("input"):
        user += "\n\n" + example["input"]
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
```

Bu, aşağıdaki gibi bir metin üretir:

```
<|im_start|>user
Overfitting nedir, açıkla.<|im_end|>
<|im_start|>assistant
Overfitting, bir modelin ezberlemesidir...<|im_end|>
```

Template'i tutturmak, insanların en sık yanlış yaptığı şeylerden biridir — uyumsuz bir format, sonuçları sessizce bozar.

### 3. LoRA adapter'larını takın

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

Bu son satır şuna benzer bir çıktı verir: `trainable params: 18,464,768 || all params: 1,562,179,072 || trainable%: 1.18`. **PEFT'in bütün olayı budur** — modelin yaklaşık %1'ini eğitiyorsunuz.

### 4. Standart `Trainer` ile train edin

Tokenize ediyoruz, padding ve label oluşturmayı bir data collator'a bırakıyoruz, ardından sıradan bir Hugging Face `Trainer` çalıştırıyoruz. Küçük bir GPU'da en çok iki ayar önemlidir:

- `optim="paged_adamw_8bit"` — memory tasarrufu için 8-bit, paged bir optimizer.
- `gradient_checkpointing=True` — backward pass'te activation'ları yeniden hesaplayarak compute karşılığında memory kazanır.

---

## Hyperparameter seçimi

| Hyperparameter | Tipik değer | Ne işe yarar |
|---|---|---|
| `r` (rank) | 8–64 | Adapter'ın kapasitesi. Daha yüksek = daha ifade gücü, daha çok parametre. 16 harika bir default'tur. |
| `lora_alpha` | 2 × `r` | Adapter'ın katkısının scaling'i. |
| `lora_dropout` | 0.0–0.1 | Adapter üzerinde regularization. |
| `learning_rate` | 1e-4 – 3e-4 | LoRA, çok az parametre eğittiği için full fine-tuning'den daha yüksek LR'leri tolere eder. |
| `epochs` | 1–3 | Çok fazla epoch → overfitting / catastrophic forgetting. |
| `batch_size` × `grad_accum` | efektif 16–64 | Küçük GPU'larda büyük efektif batch için gradient accumulation kullanın. |
| `max_seq_length` | 512–2048 | Daha uzun sequence'lar daha çok memory ister; verinizin gerektirdiği kadarına truncate edin. |

Default'larla başlayın, loss eğrisini izleyin, sonra ayarlayın.

---

## Modelinizi değerlendirmek (evaluation)

Loss'un düşmesi gereklidir ama yeterli değildir. Her zaman **gerçek generation'lara** bakın:

- **Öncesi/sonrası:** aynı prompt'u base model ve fine-tuned model üzerinde çalıştırın. Notebook bunu açıkça yapar.
- **Held-out prompt'lar:** modelin training sırasında *görmediği* girdilerde test edin. Bu, overfitting'i yakalar.
- **Format uyumu:** belirli bir çıktı formatı için fine-tune ettiyseniz, buna tutarlı şekilde uyduğunu doğrulayın.
- **Regression kontrolü:** genel yeteneğini kaybetmediğinden emin olun (*catastrophic forgetting* işareti).

Ciddi projeler için küçük bir evaluation set'i oluşturup tutarlı bir şekilde puanlayın — basit bir rubric bile tek bir örneğe göz gezdirmekten iyidir.

---

## Deployment: adapter mı, merged model mı?

LoRA ile fine-tune edilmiş bir modeli yayınlamak için iki seçeneğiniz var:

**Seçenek A — Adapter'ı yayınlayın (iterasyon için önerilir).**
Yalnızca LoRA weight'lerini (birkaç MB) kaydedin. Inference sırasında base model'i yükleyip adapter'ı PEFT ile uygulayın. Böylece memory'de tek bir base model tutarken birçok adapter'ı değiştirebilirsiniz.

```python
trainer.model.save_pretrained("my-adapter")
```

**Seçenek B — Bağımsız bir modele merge edin (production servisi için önerilir).**
LoRA güncellemesini base weight'lerin içine işleyin, böylece inference artık PEFT'e bağlı olmaz. vLLM veya TGI gibi serving engine'leri için kullanışlıdır.

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16)
merged = PeftModel.from_pretrained(base, "my-adapter").merge_and_unload()
merged.save_pretrained("merged-model")
```

Merge işlemini **full-precision** bir modele yaptığınızı, 4-bit olana değil, unutmayın — bu yüzden merge, training'den daha fazla memory ister.

---

## Sık yapılan hatalar

- **Yanlış chat template.** Training formatınız modelin beklediğiyle eşleşmezse sonuçlar sessizce bozulur. Her zaman `tokenizer.apply_chat_template` kullanın.
- **Gradient checkpointing ile `use_cache = False` yapmayı unutmak.** Bunlar çakışır; notebook, training sırasında KV cache'i devre dışı bırakır ve generation için tekrar açar.
- **Çok düşük learning rate.** LoRA, full fine-tuning'den *daha yüksek* bir LR ister. Hiçbir şey değişmiyor gibiyse, LR'yi yükseltin.
- **Çok fazla epoch.** Daha fazlası daha iyi değildir. Modelin training verisini papağan gibi tekrarlamasına veya genel becerilerini kaybetmesine dikkat edin.
- **Padding token'ın ayarlanmaması.** Birçok tokenizer'da `pad_token` açıkça ayarlanmalıdır (biz `eos_token`'a geri düşüyoruz).
- **4-bit modeli merge etmek.** Merge'den önce her zaman fp16/bf16 olarak yeniden yükleyin, yoksa kaliteyi düşürürsünüz.

---

## Buradan sonrası

- **Veriyi büyütün.** Bu hızlı başlangıç, hız için 2.000 örnek kullanır. Gerçek kalite için daha fazlasını kullanın.
- **`trl`'in `SFTTrainer`'ını deneyin.** Bu boilerplate'in çoğunu, completion-only loss masking dahil, halleden daha üst seviye bir trainer.
- **Prompt'u mask'leyin (completion-only training).** Loss'u yalnızca assistant'ın yanıtı üzerinde hesaplamak, instruction following'i çoğu zaman iyileştirir.
- **Daha büyüğe geçin.** QLoRA, 7B–13B modelleri tek bir 16–24 GB GPU üzerinde fine-tune etmenize izin verir.
- **Diğer PEFT yöntemlerini keşfedin.** DoRA, IA³ ve prompt tuning, `peft` kütüphanesinde tek bir config değişikliği uzağınızdadır.

---

### Tek paragraflık özet

**Fine-tuning**, pre-trained bir modeli kendi verinizde eğitmeye devam etmektir. Bunu tamamıyla (full) yapmak fahiş derecede pahalıdır; bu yüzden parametrelerin küçük bir dilimini eğitmek için **PEFT** kullanırız. En popüler PEFT yöntemi **LoRA**'dır; modeli dondurur ve seçili katmanlar için küçük bir low-rank güncelleme (`B·A`) öğrenir. **QLoRA** bir adım daha ileri gider ve frozen base'i **4-bit** olarak saklar; gerçek modelleri **ücretsiz bir Colab GPU'su** üzerinde fine-tune etmenizi mümkün kılan da budur. Notebook'u açın, baştan sona çalıştırın ve dakikalar içinde fine-tune edilmiş bir modeliniz olsun.

*Şimdi notebook'u açın ve bir şeyler train edin.*
